from src.LCA.lca_cbcl import perform_lca
from src.preprocess.prepare_data import prepare_data


def main():
    # Prepare the data for the specified wave
    prepare_data(version_name="test")

    # Perform LCA on the prepared data
    perform_lca(
        version_name="test",
        num_blrt_repetitions=10,
    )


if __name__ == "__main__":
    main()
