#[cxx::bridge(namespace = "bibiocr")]
pub mod bridge {
    struct PipelineResponse {
        markdown: String,
        layout_path: String,
        output_dir: String,
    }

    unsafe extern "C++" {
        include!("bibiocr/bridge.hpp");

        fn run_pipeline(
            image_path: &CxxString,
            output_dir: &CxxString,
            config_path: &CxxString,
        ) -> Result<PipelineResponse>;
    }
}
