import sys

import pandas as pd
from loguru import logger


def read_raw_excel(file_path: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Reads raw .xlsx file.

    Args:
        file_path (str): path to a file.
        sheet_name (str, optional): name of the sheet. Defaults to `None`.

    Returns:
        pd.DataFrame: DataFrame of read file.
    """
    return pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name)


def read_raw_csv(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Reads raw .csv file.

    Args:
        file_path (str): path to a file.
        sep (str, optional): data separator. Defaults to `,`.

    Returns:
        pd.DataFrame: DataFrame of read file.
    """
    return pd.read_csv(file_path, sep=sep)


def read_raw_file(file_path: str, sep: str = ",", sheet_name: str | None = None) -> pd.DataFrame:
    """Reads raw .xslx or .csv file.

    Args:
        file_path (str): path to a file.
        sep (str, optional): data separator, used when reading .csv file. Defaults to `,`.
        sheet_name (str, optional): name of the sheet, used when reading .xlsx files. Defaults to `None`.

    Returns:
        pd.DataFrame: DataFrame of read file.
    """
    file_extension = file_path.split(".")[-1]

    logger.info(f"Reading a file: {file_path}.")
    if file_extension == "xlsx":
        return read_raw_excel(file_path, sheet_name)
    if file_extension == "csv":
        return read_raw_csv(file_path, sep)

    logger.error(f"Invalid file format: `{file_extension}`, only `csv` and `xlsx` are possible.")
    sys.exit(1)


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame read from an Excel file.

    Args:
        df (pd.DataFrame): a DataFrame with invalid column names.

    Returns:
        pd.DataFrame: a DataFrame with fixed column names.
    """
    logger.info("Fixing colum names.")
    good_col_names = [col.replace("\xa0", "") for col in df.columns]
    good_col_names = [col.strip() for col in good_col_names]
    good_col_names = [col.lower() for col in good_col_names]

    df.columns = good_col_names

    return df


def filter_columns(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    """Filters DataFrame by keeping columns specified in the `keep_cols`.

    Args:
        df (pd.DataFrame): a DataFrame to filter.
        keep_cols (list): list of column names that will remain in the DataFrame.

    Returns:
        pd.DataFrame: filtered DataFrame.
    """
    if keep_cols is None:
        logger.warning(
            "`keep_cols` keyword was given in the configuration file, but it's empty. Proceeding without changes.",
        )
        return df

    if not all(column in df.columns for column in keep_cols):
        logger.error(
            f"Columns {set(keep_cols) - set(df.columns)} are given in the `keep_cols`, "
            "but they are not in the dataframe.",
        )
        sys.exit(1)

    logger.info(f"From a dataframe keeping only `{keep_cols}` columns.")
    return df[keep_cols]


def create_series(df: pd.DataFrame, new_col_template: str, formula_template: str) -> pd.DataFrame:
    """Creates multiple columns with names based on the `new_col_template` and calculated using `formula_template`.
    For example if we want to change seconds to minutes in our data. Let's assume that we have multiple columns
    named `["sec1", "sec2", "sec3", ...]`. Using `new_col_template = "minN"` and `formula_template = "df.secN/60"`
    we create new columns: `["min1", "min2", "min3", ...]` where values are corresponding to seconds and divided by 60.
    We don't have to know total number of columns. Also it is possible to use more than one column in the formula.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        new_col_template (str): new column names template, it must contain `N` somewhere.
        formula_template (str): formula template that will be used to create new columns, it must contain `N` somewhere.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if "N" not in new_col_template:
        logger.error(f"Incorrect new column naming template, this `{new_col_template}` must contain `N`.")
        sys.exit(1)

    if "N" not in formula_template:
        logger.error(f"Incorrect formula template, this `{formula_template}` must contain `N`.")
        sys.exit(1)

    logger.info(f"Creating new multiple columns: `{new_col_template}` using formula `{formula_template}`.")
    idx = 0

    while True:
        idx += 1
        new_col_name = new_col_template.replace("N", str(idx))
        formula = formula_template.replace("N", str(idx))

        if idx == 1:
            try:
                eval(formula)
            except AttributeError:
                logger.error(logger.error(f"Check if the formula is correct: `{formula}`."))
                sys.exit(1)

        try:
            df[new_col_name] = eval(formula)
        except AttributeError:
            break

    logger.info(f"Created {idx-1} new columns `{new_col_template} = {formula_template}`.")
    return df


def create_multiple_cols(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Creates multiple series of new columns based on the dictionary. For detailed documentation look for
    `create_series` function that is used in this one.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): configuration dictionary with new column names template and formula template.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        logger.warning(
            "`multiple_cols` keyword was given in the configuration file, but it's empty. Proceeding without changes.",
        )
        return df

    for new_col, formula in cfg.items():
        df = create_series(df, new_col, formula)

    return df


def create_single_cols(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create a single column based on the template given in the configuration file.
    For example if we want to add a frequency to data that contains period, in the configuration file,
    we need to set `cfg={"frequency":"1/period"}`. Column named "period" must exist in the `df`.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): dictionary that contains {new_column_name}:{formula}

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        logger.warning(
            "`single_cols` keyword was given in the configuration file, but it's empty. Proceeding without changes.",
        )
        return df

    for new_col, formula in cfg.items():
        logger.info(f"Creating a new columns: `{new_col}` based on the formula `{formula}`.")
        try:
            df[new_col] = eval(formula)
        except AttributeError:
            logger.error(f"Check if the formula is correct: `{formula}`.")
            sys.exit(1)

    return df


def create_new_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Creates new columns based on the given formulas. It is possible to create or more single columns using
    keyword `single_cols`. It is also possible to add multiple series of `N` new columns using keyword `multiple_cols`.
    How does each of them work look for `create_single_cols` and `create_series` functions.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): configuration file that contains keywords `single_cols` and/or `multiple_cols`.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        return df

    for new_cols_type, cfg_new_columns in cfg.items():
        if new_cols_type == "single_cols":
            df = create_single_cols(df, cfg_new_columns)

        elif new_cols_type == "multiple_cols":
            df = create_multiple_cols(df, cfg_new_columns)

        else:
            logger.error(
                f"Invalid entry in `create_new_cols`. `{new_cols_type}` was given,"
                f"but only `single_cols` and `multiple_cols` are possible.",
            )
            sys.exit(1)

    return df


def load_dataset(cfg: dict, sheet_name: str | None = None) -> pd.DataFrame:
    """Loads and preprocess dataset based on the dictionary.

    Args:
        cfg (dict): a dictionary with a path, colum names, transformations, separator, etc.
        sheet_name (str, optional): name of the sheet when read from .xlsx file.

    Returns:
        pd.DataFrame: a DataFrame made of read file.
    """
    if sheet_name is None:
        sheet_name = cfg.get("sheet_name")

    df = read_raw_file(
        cfg.get("file_path"),
        "," if cfg.get("data_sep") is None else cfg.get("data_sep"),
        sheet_name,
    )
    df = fix_column_names(df)
    if cfg.get("keep_cols") is not None:
        df = filter_columns(df, cfg["keep_cols"])
    df = create_new_columns(df, cfg.get("create_new_cols"))

    logger.info(f"Reading data successful, a dataframe has a shape: {df.shape}.")

    return df
