This `resources` directory contains files which will be downloaded when requested by the `ensure_data_file()` function which uses `pooch`.

I strongly suggest placing all files downloaded from the GitHub resources directory into a `data_cache` directory on the local machine. As the name suggests, the `data_cache` directory should be considered as temporary. It can be deleted and rebuilt as files are downloaded when needed. Note that `data_cache` is included in 
`.gitignore` to prevent duplication of files in the repository.
