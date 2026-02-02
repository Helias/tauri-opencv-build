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
        let out_dir = env::var("OUT_DIR").unwrap();
        let target_dir = PathBuf::from(&out_dir)
          .parent()
          .and_then(|p| p.parent())
          .and_then(|p| p.parent())
          .unwrap()
          .to_path_buf();

        println!("cargo:warning=Copying OpenCV DLLs from {:?} to {:?}", dll_source, target_dir);

        // Copy all OpenCV-related DLLs
        if let Ok(entries) = fs::read_dir(&dll_source) {
          for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
              let filename_str = filename.to_string_lossy();
              // Copy OpenCV DLLs and their dependencies
              if filename_str.contains("opencv") ||
                 filename_str.contains("tbb") ||
                 filename_str.contains("jpeg") ||
                 filename_str.contains("png") ||
                 filename_str.contains("webp") ||
                 filename_str.contains("zlib") {
                let dest = target_dir.join(filename);
                if let Err(e) = fs::copy(&path, &dest) {
                  println!("cargo:warning=Failed to copy {}: {}", filename_str, e);
                } else {
                  println!("cargo:warning=Copied {}", filename_str);
                }
              }
            }
          }
        }
      } else {
        println!("cargo:warning=VCPKG_ROOT is set but DLL directory not found at {:?}", dll_source);
      }
    } else {
      println!("cargo:warning=VCPKG_ROOT not set, skipping OpenCV DLL copy");
    }
  }
}
