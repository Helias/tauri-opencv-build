#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

mod annotate_file;
use annotate_file::{annotate_video, ProcessingState, PROCESSING_STATE};

#[tauri::command]
async fn hello_world_command(_app: tauri::AppHandle) -> Result<String, String> {
  println!("I was invoked from JS!");
  Ok("Hello world from Tauri!".into())
}

#[tauri::command]
async fn process_file(app: tauri::AppHandle, file_path: String) -> Result<String, String> {
  println!("Processing file: {}", file_path);

  // Start processing in a separate thread
  let app_clone = app.clone();
  let file_path_clone = file_path.clone();

  std::thread::spawn(move || {
    // Create a temporary output path
    let temp_dir = std::env::temp_dir();
    let output_filename = format!("swishai_output_{}.mp4", chrono::Local::now().timestamp());
    let output_path = temp_dir.join(&output_filename);
    let output_path_str = output_path.to_string_lossy().to_string();

    println!("Temporary output: {}", output_path_str);

    match annotate_video(app_clone, file_path_clone.clone(), output_path_str.clone()) {
      Ok(_) => {
        println!("✅ Processing completed successfully");
      },
      Err(e) => {
        eprintln!("❌ Error processing video: {}", e);
        // Update state to error
        if let Ok(mut state) = PROCESSING_STATE.lock() {
          state.status = "error".to_string();
          state.error = Some(format!("Error processing video: {}", e));
        }
      }
    }
  });

  Ok("Processing started".into())
}

#[tauri::command]
async fn get_processing_progress() -> Result<ProcessingState, String> {
  match PROCESSING_STATE.lock() {
    Ok(state) => Ok(state.clone()),
    Err(e) => Err(format!("Failed to get processing state: {}", e)),
  }
}

#[tauri::command]
async fn copy_file(source_path: String, dest_path: String) -> Result<(), String> {
  std::fs::copy(&source_path, &dest_path)
    .map(|_| ())
    .map_err(|e| format!("Failed to copy file: {}", e))
}

fn main() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_http::init())
    .plugin(tauri_plugin_fs::init())
    .invoke_handler(tauri::generate_handler![hello_world_command, process_file, get_processing_progress, copy_file])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
