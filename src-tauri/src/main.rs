#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rfd::FileDialog;
use std::path::Path;
use std::process::Command;

#[tauri::command]
fn pick_video_files() -> Vec<String> {
    FileDialog::new()
        .add_filter("Videos", &["mp4", "mov", "avi", "webm"])
        .set_title("Select one or more videos")
        .pick_files()
        .unwrap_or_default()
        .into_iter()
        .map(|path| path.display().to_string())
        .collect()
}

#[tauri::command]
fn pick_directory() -> Option<String> {
    FileDialog::new()
        .set_title("Select a folder")
        .pick_folder()
        .map(|path| path.display().to_string())
}

#[tauri::command]
fn pick_directories() -> Vec<String> {
    FileDialog::new()
        .set_title("Select one or more folders")
        .pick_folders()
        .unwrap_or_default()
        .into_iter()
        .map(|path| path.display().to_string())
        .collect()
}

#[tauri::command]
fn pick_ocio_file() -> Option<String> {
    FileDialog::new()
        .add_filter("OpenColorIO", &["ocio"])
        .set_title("Select an OCIO configuration")
        .pick_file()
        .map(|path| path.display().to_string())
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let target = Path::new(&path);
    if !target.exists() {
        return Err(format!("Path does not exist: {}", path));
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(target)
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(target)
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            pick_video_files,
            pick_directory,
            pick_directories,
            pick_ocio_file,
            reveal_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running ReMap desktop shell");
}
