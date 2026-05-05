import fire
from data_routines.dist_data_processing import main as dist_data_processing_main  # noqa
from batch_inference import batch_inference  # noqa


def main():
    fire.Fire(
        {
            "dist_data_processing": dist_data_processing_main,
            "batch_inference": batch_inference,
        }
    )


if __name__ == "__main__":
    main()
