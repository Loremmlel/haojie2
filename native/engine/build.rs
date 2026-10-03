// 构建身份覆盖实际源文件和规则包，不依赖 Git、Node 或开发机路径。
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

fn collect(path: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(path).expect("source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            collect(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    let mut files = vec!["Cargo.toml".into(), "Cargo.lock".into(), "build.rs".into()];
    collect(Path::new("src"), &mut files);
    collect(Path::new("data"), &mut files);
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.to_str().expect("UTF-8 source path").replace('\\', "/");
        let bytes = fs::read(&path).expect("source bytes");
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    println!("cargo:rustc-env=HAOJIE_SOURCE_SHA256={:x}", hash.finalize());
    for name in ["TARGET", "PROFILE"] {
        println!(
            "cargo:rustc-env=HAOJIE_{name}={}",
            std::env::var(name).unwrap()
        );
    }
    let rustc = Command::new(std::env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .expect("rustc identity");
    assert!(rustc.status.success());
    println!(
        "cargo:rustc-env=HAOJIE_RUSTC={}",
        String::from_utf8(rustc.stdout).unwrap().trim()
    );
}
