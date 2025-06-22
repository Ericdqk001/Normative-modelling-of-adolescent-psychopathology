from src.LCA.lca_cbcl import perform_lca
from src.preprocess.prepare_data import prepare_data
from src.preprocess.split import split


def main(version_name: str = "test"):
    # Prepare the data for the specified wave
    prepare_data(version_name=version_name)

    # Perform LCA on the prepared data
    perform_lca(
        version_name=version_name,
        num_blrt_repetitions=1,
    )

    # Split the data into sets
    split(
        version_name=version_name,
    )


if __name__ == "__main__":
    version_name = "test"
    main(
        version_name=version_name,
    )
