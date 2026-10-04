use std::{env, path::PathBuf};

fn main() {
    const JAVA: &str = "src/android/CameraFragment.java";
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("android") {
        return;
    }
    println!("cargo:rerun-if-changed={JAVA}");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let android_jar = android_build::android_jar(None).expect("android.jar");
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join(JAVA);
    assert!(
        android_build::JavaBuild::new()
            .class_path(android_jar.clone())
            .classes_out_dir(out.clone())
            .file(source)
            .java_source_version(17)
            .java_target_version(17)
            .compile()
            .expect("javac")
            .success(),
        "camera Java helper failed to compile"
    );
    let class = out.join("com/bibiocr/mobile/CameraFragment.class");
    let d8 = android_build::android_d8_jar(None).expect("d8.jar");
    assert!(
        android_build::JavaRun::new()
            .class_path(d8)
            .main_class("com.android.tools.r8.D8")
            .arg("--classpath")
            .arg(android_jar)
            .arg("--min-api")
            .arg("26")
            .arg("--output")
            .arg(&out)
            .arg(class)
            .run()
            .expect("d8")
            .success(),
        "camera DEX helper failed to compile"
    );
}
