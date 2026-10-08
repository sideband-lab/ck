use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

// Import from the main.rs module
use ck_search::{CkMcpServer, HybridSearchRequest, RegexSearchRequest, SemanticSearchRequest};

#[tokio::test]
async fn test_mcp_semantic_search_basic_functionality() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // Test first page request
    let request = SemanticSearchRequest {
        query: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(10),
        threshold: Some(0.0),
        cursor: None,
        page_size: Some(5),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let (summary, response) = server
        .handle_semantic_search(request, None, None)
        .await
        .expect("MCP semantic search should succeed with a real embedding model");
    assert!(summary.contains("Page 1"));
    assert_eq!(response["search"]["mode"], "semantic");
    assert!(response["search"].is_object());
    assert!(response["results"]["matches"].is_array());
    assert!(response["pagination"].is_object());
    assert!(response["pagination"]["current_page"].is_number());
    assert!(response["results"]["count"].is_number());
    assert!(response["results"]["has_more"].is_boolean());
}

#[tokio::test]
async fn test_mcp_regex_search_basic_functionality() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    let request = RegexSearchRequest {
        pattern: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        ignore_case: Some(true),
        context: Some(2),
        cursor: None,
        page_size: Some(3),
        include_snippet: Some(true),
        snippet_length: Some(50),
        ..Default::default()
    };

    let result = server.handle_regex_search(request).await;

    if let Ok((summary, response)) = result {
        assert!(summary.contains("Page 1"));

        // Verify response structure matches expected format
        assert_eq!(response["search"]["mode"], "regex");
        assert!(response["results"]["matches"].is_array());
        assert!(response["pagination"]["page_size"].is_number());

        // Check match structure for regex search
        if let Some(matches) = response["results"]["matches"].as_array()
            && !matches.is_empty()
        {
            let first_match = &matches[0];
            assert_eq!(first_match["type"], "regex_match");
            assert!(first_match["match"]["line_number"].is_number());
        }
    }
}

#[tokio::test]
async fn test_mcp_hybrid_search_basic_functionality() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    let request = HybridSearchRequest {
        query: "function error".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(5),
        threshold: Some(0.01),
        cursor: None,
        page_size: Some(2),
        include_snippet: Some(true),
        snippet_length: Some(200),
        context_lines: Some(1),
        ..Default::default()
    };

    let (summary, response) = server
        .handle_hybrid_search(request)
        .await
        .expect("MCP hybrid search should succeed");
    assert!(summary.contains("Page 1"));
    assert_eq!(response["search"]["mode"], "hybrid");
    let matches = response["results"]["matches"].as_array().unwrap();
    assert!(!matches.is_empty(), "expected semantic or lexical matches");
    assert_eq!(matches[0]["type"], "hybrid_match");
    assert!(matches[0]["match"]["score"].is_number());
    assert!(matches[0]["match"]["rrf_score"].is_number());
}

#[tokio::test]
async fn test_mcp_invalid_cursor_handling() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // Test with invalid cursor
    let request = SemanticSearchRequest {
        query: "test".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(10),
        threshold: Some(0.1),
        cursor: Some("invalid_cursor".to_string()),
        page_size: Some(5),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let result = server.handle_semantic_search(request, None, None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_mcp_search_parameters_validation() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // Test with extreme page size (should be clamped)
    let request = SemanticSearchRequest {
        query: "test".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(10),
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(1000), // Should be clamped to 200
        include_snippet: Some(true),
        snippet_length: Some(10000), // Should be clamped to 2000
        context_lines: Some(50),     // Should be clamped to 10
        ..Default::default()
    };

    let (_, response) = server
        .handle_semantic_search(request, None, None)
        .await
        .expect("MCP parameter validation should return a response");
    let page_size = response["pagination"]["page_size"].as_u64().unwrap();
    assert_eq!(page_size, 200);
}

#[tokio::test]
async fn test_mcp_nonexistent_path() {
    let server = CkMcpServer::new(PathBuf::from("/nonexistent")).unwrap();

    let request = SemanticSearchRequest {
        query: "test".to_string(),
        path: "/definitely/does/not/exist".to_string(),
        top_k: Some(10),
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(5),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let result = server.handle_semantic_search(request, None, None).await;
    assert!(result.is_err());

    if let Err(error) = result {
        let msg = error.to_string();
        assert!(
            msg.contains("does not exist"),
            "expected 'does not exist' in error, got: {msg}"
        );
    }
}

async fn create_test_files() -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create some test files with different content
    let files = vec![
        (
            "test1.rs",
            "function main() {\n    println!(\"Hello world\");\n}\n\nfunction helper() {\n    // Some helper code\n}",
        ),
        (
            "test2.js",
            "function calculate(x, y) {\n    return x + y;\n}\n\nfunction error_handler() {\n    console.error(\"An error occurred\");\n}",
        ),
        (
            "test3.py",
            "def process_data(data):\n    try:\n        return data.process()\n    except Exception as e:\n        handle_error(e)\n\ndef handle_error(error):\n    print(f\"Error: {error}\")",
        ),
        (
            "test4.txt",
            "This is a text file with some content.\nIt contains various words and phrases.\nSome lines mention functions and errors.",
        ),
    ];

    for (filename, content) in files {
        let file_path = temp_dir.path().join(filename);
        fs::write(file_path, content)
            .await
            .expect("Failed to write test file");
    }

    temp_dir
}

#[tokio::test]
async fn test_mcp_top_k_page_size_interaction() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // Test case 1: top_k=5, page_size=3 should give us one page with 3 results, then one with 2
    let request = SemanticSearchRequest {
        query: "test".to_string(), // Should match content in test files
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(5),
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(3),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let (summary, response) = server
        .handle_semantic_search(request, None, None)
        .await
        .expect("first semantic page should succeed");
    let matches = response["results"]["matches"].as_array().unwrap();
    assert!(matches.len() <= 3);
    assert!(response["results"]["total_count"].as_u64().unwrap() <= 5);
    assert!(summary.contains("top_k: 5"));

    // Test case 2: top_k=2, page_size=10 should give us one page with 2 results max
    let request2 = SemanticSearchRequest {
        query: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(2),
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(10),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let (summary2, response2) = server
        .handle_semantic_search(request2, None, None)
        .await
        .expect("second semantic page should succeed");
    assert!(response2["results"]["total_count"].as_u64().unwrap() <= 2);
    assert!(response2["results"]["matches"].as_array().unwrap().len() <= 2);
    assert!(summary2.contains("top_k: 2"));
}

#[tokio::test]
async fn test_mcp_semantic_search_with_missing_files() {
    use std::fs;

    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // First, do an initial search to ensure the index is created
    let initial_request = SemanticSearchRequest {
        query: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(10),
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(10),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    server
        .handle_semantic_search(initial_request, None, None)
        .await
        .expect("initial indexing/search should succeed");

    // Now remove one of the test files to simulate a stale index
    let file_to_remove = temp_dir.path().join("test1.rs");
    if file_to_remove.exists() {
        fs::remove_file(&file_to_remove).expect("Failed to remove test file");
    }

    // Search again - this should handle the missing file gracefully
    let request = SemanticSearchRequest {
        query: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        top_k: Some(20), // Request more results to increase chance of hitting missing file
        threshold: Some(0.1),
        cursor: None,
        page_size: Some(10),
        include_snippet: Some(true),
        snippet_length: Some(100),
        context_lines: Some(0),
        ..Default::default()
    };

    let (summary, response) = server
        .handle_semantic_search(request, None, None)
        .await
        .expect("semantic search should tolerate a stale sidecar");
    assert!(response["results"]["matches"].is_array());
    assert!(summary.contains("top_k: 20"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_mcp_index_status_reports_real_index_size() {
    use ck_search::IndexStatusRequest;
    // This test asserts the in-tree `.ck` size, so pin default behavior even if
    // CK_INDEX_DIR is set in the ambient environment.
    unsafe { std::env::remove_var("CK_INDEX_DIR") };

    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    // Build the index by running a semantic search first (auto-indexes)
    let search = SemanticSearchRequest {
        query: "error handling".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        threshold: Some(0.0),
        ..Default::default()
    };
    server
        .handle_semantic_search(search, None, None)
        .await
        .expect("semantic search should succeed");

    let request = IndexStatusRequest {
        path: temp_dir.path().to_string_lossy().to_string(),
    };
    let (_, response) = server
        .handle_index_status(request, None, None)
        .await
        .expect("index_status should succeed");

    let info = &response["index_status"];
    assert_eq!(info["index_exists"], true);

    // index_size_bytes must be the recursive on-disk size of .ck, not the
    // directory entry's own size. Compare against an independent walk.
    let expected_size: u64 = walkdir::WalkDir::new(temp_dir.path().join(".ck"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    assert!(expected_size > 0, "test setup: index should not be empty");
    assert_eq!(info["index_size_bytes"], expected_size);

    // Chunk accounting must be present and consistent
    let total = info["total_chunks"].as_u64().unwrap();
    let embedded = info["embedded_chunks"].as_u64().unwrap();
    let pending = info["chunks_pending_embedding"].as_u64().unwrap();
    assert_eq!(pending, total - embedded);
}

#[tokio::test]
async fn test_mcp_search_time_excludes_indexing_and_reports_it() {
    let temp_dir = create_test_files().await;
    let server = CkMcpServer::new(temp_dir.path().to_path_buf()).unwrap();

    let request = || SemanticSearchRequest {
        query: "error handling".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        threshold: Some(0.0),
        ..Default::default()
    };

    // First search on an unindexed directory triggers auto-indexing,
    // which must be reported separately from search time.
    let (_, response) = server
        .handle_semantic_search(request(), None, None)
        .await
        .expect("semantic search should succeed");
    let indexing = &response["metadata"]["indexing"];
    assert!(
        indexing.is_object(),
        "semantic search should report indexing metadata, got: {indexing}"
    );
    assert_eq!(indexing["triggered"], true);
    assert!(indexing["files_indexed"].as_u64().unwrap() > 0);
    assert!(response["metadata"]["search_time_ms"].is_number());

    // Second search finds a fresh index: the staleness check runs but no work
    let (_, response) = server
        .handle_semantic_search(request(), None, None)
        .await
        .expect("semantic search should succeed");
    let indexing = &response["metadata"]["indexing"];
    assert_eq!(indexing["triggered"], false);
    assert_eq!(indexing["files_indexed"], 0);

    // Regex search never touches the index
    let regex_request = RegexSearchRequest {
        pattern: "function".to_string(),
        path: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    let (_, response) = server
        .handle_regex_search(regex_request)
        .await
        .expect("regex search should succeed");
    assert!(response["metadata"]["indexing"].is_null());
}
