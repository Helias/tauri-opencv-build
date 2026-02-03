use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
  tauri_build::build();

  // Copy OpenCV DLLs on Windows
  #[cfg(target_os = "windows")]
  {
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");

    if let Ok(vcpkg_root) = env::var("VCPKG_ROOT") {
      let dll_source = PathBuf::from(&vcpkg_root)
        .join("installed")
        .join("x64-windows")
        .join("bin");

      if dll_source.exists() {
        // Create a libs directory in src-tauri for Tauri to bundle
        let libs_dir = PathBuf::from("libs");
        fs::create_dir_all(&libs_dir).ok();

        println!("cargo:warning=Copying DLLs from {:?} to {:?}", dll_source, libs_dir);

        // Copy all OpenCV-related DLLs and dependencies
        if let Ok(entries) = fs::read_dir(&dll_source) {
          for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
              let filename_str = filename.to_string_lossy();
              // Copy OpenCV DLLs and their dependencies, including ONNX Runtime and DirectML
              if filename_str.contains("opencv") ||
                 filename_str.contains("tbb") ||
                 filename_str.contains("jpeg") ||
                 filename_str.contains("png") ||
                 filename_str.contains("webp") ||
                 filename_str.contains("zlib") ||
                 filename_str.contains("onnxruntime") ||
                 filename_str.contains("directml") {
                let dest = libs_dir.join(filename);
                if let Err(e) = fs::copy(&path, &dest) {
                  println!("cargo:warning=Failed to copy {}: {}", filename_str, e);
                } else {
                  println!("cargo:warning=Copied {} to libs/", filename_str);
                }
              }
            }
          }
        }

        // Tell Tauri where to find the DLLs
        println!("cargo:rustc-link-search=native={}", libs_dir.canonicalize().unwrap().display());
      } else {
        println!("cargo:warning=VCPKG_ROOT is set but DLL directory not found at {:?}", dll_source);
      }
    } else {
      println!("cargo:warning=VCPKG_ROOT not set, skipping OpenCV DLL copy");
    }
  }
}
