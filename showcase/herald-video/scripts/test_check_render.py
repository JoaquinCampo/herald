from check_render import validate_probe


def test_final_probe_contract() -> None:
    validate_probe(
        {
            "format": {"duration": "85.000000"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
    )
