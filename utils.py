import os
import glob
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import kneed
import scipy
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random


def load_data(plate, filetype):
    """load all data from a single experiment into a single dataframe"""
    #path = os.path.join("../profiles", f"{exp}", f"{plate}", f"*_{filetype}")
    path = os.path.join("../benchmark/data", f"{plate}_{filetype}")
    files = glob.glob(path)
    #print(files)
    df = pd.concat(pd.read_csv(_, low_memory=False) for _ in files)
    return df


def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of juget_featurecolsst metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]


def remove_negcon_and_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=["Metadata_broad_sample"])
        .reset_index(drop=True)
    )
    return df


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = df.dropna(subset=["Metadata_broad_sample"]).reset_index(drop=True)
    return df


def concat_profiles(df1, df2):
    """Concatenate dataframes"""
    if df1.shape[0] == 0:
        df1 = df2.copy()
    else:
        frames = [df1, df2]
        df1 = pd.concat(frames, ignore_index=True, join="inner")

    return df1


def create_replicability_df(
    replicability_map_df, replicability_fp_df, metric, modality, cell, timepoint
):
    _replicability_map_df = replicability_map_df
    _replicability_fp_df = replicability_fp_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _fp_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fp": f"{metric.fp:.3f}",
        },
        index=[len(_replicability_fp_df)],
    )
    _replicability_fp_df = concat_profiles(_replicability_fp_df, _fp_df)

    _map_df = metric.map.copy()
    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _replicability_map_df = concat_profiles(_replicability_map_df, _map_df)

    _replicability_fp_df["fp"] = _replicability_fp_df["fp"].astype(float)
    _replicability_map_df["mAP"] = _replicability_map_df["mAP"].astype(float)

    return _replicability_map_df, _replicability_fp_df


def create_matching_df(
    matching_map_df, matching_fp_df, metric, modality, cell, timepoint
):
    _matching_map_df = matching_map_df
    _matching_fp_df = matching_fp_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _fp_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fp": f"{metric.fp:.3f}",
        },
        index=[len(_matching_fp_df)],
    )
    _matching_fp_df = concat_profiles(_matching_fp_df, _fp_df)

    _map_df = metric.map.copy()
    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _matching_map_df = concat_profiles(_matching_map_df, _map_df)

    _matching_fp_df["fp"] = _matching_fp_df["fp"].astype(float)
    _matching_map_df["mAP"] = _matching_map_df["mAP"].astype(float)

    return _matching_map_df, _matching_fp_df


def create_gene_compound_matching_df(
    gene_compound_matching_map_df,
    gene_compound_matching_fp_df,
    metric,
    modality_1,
    modality_2,
    cell,
    timepoint1,
    timepoint2,
):
    _gene_compound_matching_map_df = gene_compound_matching_map_df
    _gene_compound_matching_fp_df = gene_compound_matching_fp_df

    _modality_1 = modality_1
    _modality_2 = modality_2
    _cell = cell
    _timepoint_1 = timepoint1
    _timepoint_2 = timepoint2
    _time_1 = time_point(_modality_1, _timepoint_1)
    _time_2 = time_point(_modality_2, _timepoint_2)

    _description = f"{_modality_1}_{cell}_{_time_1}-{_modality_2}_{cell}_{_time_2}"

    _fp_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality1": f"{_modality_1}_{_time_1}",
            "Modality2": f"{_modality_2}_{_time_2}",
            "Cell": _cell,
            "fp": f"{metric.fp:.3f}",
        },
        index=[len(_gene_compound_matching_fp_df)],
    )
    _gene_compound_matching_fp_df = concat_profiles(
        _gene_compound_matching_fp_df, _fp_df
    )

    _map_df = metric.map.copy()
    _map_df["Description"] = f"{_description}"
    _map_df["Modality1"] = f"{_modality_1}_{_time_1}"
    _map_df["Modality2"] = f"{_modality_2}_{_time_2}"
    _map_df["Cell"] = f"{_cell}"
    _gene_compound_matching_map_df = concat_profiles(
        _gene_compound_matching_map_df, _map_df
    )

    _gene_compound_matching_fp_df["fp"] = _gene_compound_matching_fp_df["fp"].astype(
        float
    )
    _gene_compound_matching_map_df["mAP"] = _gene_compound_matching_map_df[
        "mAP"
    ].astype(float)

    return _gene_compound_matching_map_df, _gene_compound_matching_fp_df


def consensus(profiles_df, group_by_feature):
    """
    Computes the median consensus profiles.
    Parameters:
    -----------
    profiles_df: pandas.DataFrame
        dataframe of profiles
    group_by_feature: str
        Name of the column
    Returns:
    -------
    pandas.DataFrame of the same shape as `plate`
    """

    metadata_df = get_metadata(profiles_df).drop_duplicates(subset=[group_by_feature])

    feature_cols = [group_by_feature] + get_featurecols(profiles_df)
    profiles_df = (
        profiles_df[feature_cols].groupby([group_by_feature]).median().reset_index()
    )

    profiles_df = metadata_df.merge(profiles_df, on=group_by_feature)

    return profiles_df


class PrecisionScores(object):
    """
    Calculate the precision scores for information retrieval.
    """

    def __init__(
        self,
        profile1,
        profile2,
        group_by_feature,
        mode,
        identify_perturbation_feature,
        within=False,
        anti_correlation=False,
        against_negcon=False,
    ):
        """
        Parameters:
        -----------
        profile1: pandas.DataFrame
            dataframe of profiles
        profile2: pandas.DataFrame
            dataframe of profiles
        group_by_feature: str
            Name of the feature to group by
        mode: str
            Whether compute replicability or matching
        identity_perturbation_feature: str
            Name of the feature that identifies perturbations
        within: bool, default: False
            Whether profile1 and profile2 are the same dataframe.
        anti_correlation: bool, default: False
            Whether both anti-correlation and correlation are used in the calculation.
        against_negcon: bool, default:  False
            Whether to calculate precision scores with respect to negcon.
        """
        self.sample_id_feature = "Metadata_sample_id"
        self.control_type_feature = "Metadata_control_type"
        self.feature = group_by_feature
        self.mode = mode
        self.identify_perturbation_feature = identify_perturbation_feature
        self.within = within
        self.anti_correlation = anti_correlation
        self.against_negcon = against_negcon

        self.profile1 = self.process_profiles(profile1)
        self.profile2 = self.process_profiles(profile2)

        if self.mode == "replicability":
            self.map1 = self.profile1[
                [self.feature, self.sample_id_feature, self.control_type_feature]
            ].copy()
            self.map2 = self.profile2[
                [self.feature, self.sample_id_feature, self.control_type_feature]
            ].copy()
        elif self.mode == "matching":
            self.map1 = self.profile1[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.sample_id_feature,
                    self.control_type_feature,
                ]
            ].copy()
            self.map2 = self.profile2[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.sample_id_feature,
                    self.control_type_feature,
                ]
            ].copy()

        self.corr = self.compute_correlation()
        self.truth_matrix = self.create_truth_matrix()
        self.cleanup()

        self.ap = self.calculate_average_precision_per_sample()
        self.map = self.calculate_average_precision_score_per_group(self.ap)
        self.mmap = self.calculate_mean_average_precision_score(self.map)

    def process_profiles(self, _profile):
        """
        Add sample id column to profiles.
        Parameters:
        -----------
        _profile: pandas.DataFrame
            dataframe of profiles
        Returns:
        -------
        pandas.DataFrame which includes the sample id column
        """

        _metadata_df = pd.DataFrame()
        _profile = _profile.reset_index(drop=True)
        _feature_df = get_featuredata(_profile)
        if self.mode == "replicability":
            _metadata_df = _profile[[self.feature, self.control_type_feature]]
        elif self.mode == "matching":
            _metadata_df = _profile[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.control_type_feature,
                ]
            ]
        width = int(np.log10(len(_profile))) + 1
        _perturbation_id_df = pd.DataFrame(
            {
                self.sample_id_feature: [
                    f"sample_{i:0{width}}" for i in range(len(_metadata_df))
                ]
            }
        )
        _metadata_df = pd.concat([_metadata_df, _perturbation_id_df], axis=1)
        _profile = pd.concat([_metadata_df, _feature_df], axis=1)
        return _profile

    def compute_correlation(self):
        """
        Compute correlation.
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values.
        """

        _profile1 = get_featuredata(self.profile1)
        _profile2 = get_featuredata(self.profile2)
        _sample_names_1 = list(self.profile1[self.sample_id_feature])
        _sample_names_2 = list(self.profile2[self.sample_id_feature])
        _corr = cosine_similarity(_profile1, _profile2)
        if self.anti_correlation:
            _corr = np.abs(_corr)
        _corr_df = pd.DataFrame(_corr, columns=_sample_names_2, index=_sample_names_1)
        _corr_df = self.process_self_correlation(_corr_df)
        _corr_df = self.process_negcon(_corr_df)
        return _corr_df

    def create_truth_matrix(self):
        """
        Compute truth matrix.
        Returns:
        -------
        pandas.DataFrame of binary truth values.
        """

        _truth_matrix = self.corr.unstack().reset_index()
        _truth_matrix = _truth_matrix.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature, 0], axis=1)
        _truth_matrix = _truth_matrix.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _truth_matrix["value"] = [
            len(np.intersect1d(x[0].split("|"), x[1].split("|"))) > 0
            for x in zip(
                _truth_matrix[f"{self.feature}_x"], _truth_matrix[f"{self.feature}_y"]
            )
        ]
        if self.within and self.mode == "replicability":
            _truth_matrix["value"] = np.where(
                _truth_matrix["level_0"] == _truth_matrix["level_1"],
                0,
                _truth_matrix["value"],
            )
        elif self.within and self.mode == "matching":
            _truth_matrix["value"] = np.where(
                _truth_matrix[f"{self.identify_perturbation_feature}_x"]
                == _truth_matrix[f"{self.identify_perturbation_feature}_y"],
                0,
                _truth_matrix["value"],
            )

        _truth_matrix = (
            _truth_matrix.pivot(index="level_1",columns= "level_0",values= "value")
            .reset_index()
            .set_index("level_1")
        )
        _truth_matrix.index.name = None
        _truth_matrix = _truth_matrix.rename_axis(None, axis=1)
        return _truth_matrix

    def calculate_average_precision_per_sample(self):
        """
        Compute average precision score per sample.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """
        _score = []
        for _sample in self.corr.index:
            _y_true, _y_pred = self.filter_nan(
                self.truth_matrix.loc[_sample].values, self.corr.loc[_sample].values
            )

            # compute corrected average precision
            random_baseline_ap = _y_true.sum() / len(_y_true)
            _score.append(
                average_precision_score(_y_true, _y_pred) - random_baseline_ap
            )

        _ap_sample_df = self.map1.copy()
        _ap_sample_df["ap"] = _score
        if self.against_negcon:
            _ap_sample_df = (
                _ap_sample_df.query(f'{self.control_type_feature}!="negcon"')
                .drop(columns=[self.control_type_feature])
                .reset_index(drop=True)
            )
        else:
            _ap_sample_df = _ap_sample_df.drop(
                columns=[self.control_type_feature]
            ).reset_index(drop=True)

        return _ap_sample_df

    def calculate_average_precision_score_per_group(self, precision_score):
        """
        Compute average precision score per sample group.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """
     
        _precision_group_df = (
            precision_score.groupby(str(self.feature))
            .apply(lambda x: np.mean(x))
            .reset_index()
            .rename(columns={"ap": "mAP"})
        )
        return _precision_group_df

    @staticmethod
    def calculate_mean_average_precision_score(precision_score):
        """
        Compute mean average precision score.
        Returns:
        -------
        mean average precision score.
        """

        return precision_score.mean().values[0]

    def process_negcon(self, _corr_df):
        """
        Keep or remove negcon
        Parameters:
        -----------
        _corr_df: pandas.DataFrame
            pairwise correlation dataframe
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values
        """
        _corr_df = _corr_df.unstack().reset_index()
        _corr_df["filter"] = 1
        _corr_df = _corr_df.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _corr_df = _corr_df.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)

        if self.against_negcon:
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.feature}_x"] != _corr_df[f"{self.feature}_y"],
                0,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_x"] == "negcon",
                1,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_y"] == "negcon",
                0,
                _corr_df["filter"],
            )
        else:
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_x"] == "negcon",
                0,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_y"] == "negcon",
                0,
                _corr_df["filter"],
            )

        _corr_df = _corr_df.query("filter==1").reset_index(drop=True)

        if self.mode == "replicability":
            self.map1 = (
                _corr_df[
                    ["level_1", f"{self.feature}_y", f"{self.control_type_feature}_y"]
                ]
                .copy()
                .rename(
                    columns={
                        "level_1": self.sample_id_feature,
                        f"{self.feature}_y": self.feature,
                        f"{self.control_type_feature}_y": self.control_type_feature,
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[
                    ["level_0", f"{self.feature}_x", f"{self.control_type_feature}_x"]
                ]
                .copy()
                .rename(
                    columns={
                        "level_0": self.sample_id_feature,
                        f"{self.feature}_x": self.feature,
                        f"{self.control_type_feature}_x": self.control_type_feature,
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
        elif self.mode == "matching":
            self.map1 = (
                _corr_df[
                    [
                        "level_1",
                        f"{self.identify_perturbation_feature}_y",
                        f"{self.feature}_y",
                        f"{self.control_type_feature}_y",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "level_1": self.sample_id_feature,
                        f"{self.feature}_y": self.feature,
                        f"{self.control_type_feature}_y": self.control_type_feature,
                        f"{self.identify_perturbation_feature}_y": f"{self.identify_perturbation_feature}",
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[
                    [
                        "level_0",
                        f"{self.identify_perturbation_feature}_x",
                        f"{self.feature}_x",
                        f"{self.control_type_feature}_x",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "level_0": self.sample_id_feature,
                        f"{self.feature}_x": self.feature,
                        f"{self.control_type_feature}_x": self.control_type_feature,
                        f"{self.identify_perturbation_feature}_x": f"{self.identify_perturbation_feature}",
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )

        _corr_df = (
            _corr_df.pivot(index="level_1",columns= "level_0",values= 0).reset_index().set_index("level_1")
        )
        _corr_df.index.name = None
        _corr_df = _corr_df.rename_axis(None, axis=1)
        return _corr_df

    @staticmethod
    def filter_nan(_y_true, _y_pred):
        """
        Filter out nan values from y_true and y_pred
        Parameters:
        -----------
        _y_true: np.array of truth values
        _y_pred: np.array of predicted values
        Returns:
        --------
        _y_true: np.array of truth values
        _y_pred: np.array of predicted values
        """
        arg = np.argwhere(~np.isnan(_y_pred))
        return _y_true[arg].flatten(), _y_pred[arg].flatten()

    def process_self_correlation(self, corr):
        """
        Process self correlation values (correlation between the same profiles)
        Parameters:
        -----------
        corr: pd.DataFrame of correlation values
        Returns:
        --------
        _corr: pd.DataFrame of correlation values
        """
        _corr = corr.unstack().reset_index().rename(columns={0: "corr"})
        _corr = _corr.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _corr = _corr.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        if self.within and self.mode == "replicability":
            _corr["corr"] = np.where(
                _corr["level_0"] == _corr["level_1"], np.nan, _corr["corr"]
            )
        elif self.within and self.mode == "matching":
            _corr["corr"] = np.where(
                _corr[f"{self.identify_perturbation_feature}_x"]
                == _corr[f"{self.identify_perturbation_feature}_y"],
                np.nan,
                _corr["corr"],
            )

        _corr = (
            _corr.pivot(index="level_1", columns="level_0", values="corr").reset_index().set_index("level_1")
        )
        _corr.index.name = None
        _corr = _corr.rename_axis(None, axis=1)

        return _corr

    def cleanup(self):
        """
        Remove rows and columns that are all NaN
        """
        keep = list((self.truth_matrix.sum(axis=1) > 0))
        self.corr["keep"] = keep
        self.map1["keep"] = keep
        self.truth_matrix["keep"] = keep

        self.corr = self.corr.loc[self.corr.keep].drop(columns=["keep"])
        self.map1 = self.map1.loc[self.map1.keep].drop(columns=["keep"])
        self.truth_matrix = self.truth_matrix.loc[self.truth_matrix.keep].drop(
            columns=["keep"]
        )


def time_point(modality, time_point):
    """
    Convert time point in hr to long or short time description
    Parameters:
    -----------
    modality: str
        perturbation modality
    time_point: int
        time point in hr
    Returns:
    -------
    str of time description
    """
    ###############################################
    ##############################################
    if modality == "compound":
        if time_point == 24:
            time = "short"
        else:
            #time = "long"
            time='ALL_EQUAL'
    elif modality == "orf":
        if time_point == 48:
            time = "short"
        else:
            #time = "long"
            time='ALL_EQUAL'
    else:
        if time_point == 96:
            time = "short"
        else:
            #####time = "long"
            time='ALL_EQUAL'

    return time


def convert_pvalue(pvalue):
    """
    Convert p value format
    Parameters:
    -----------
    pvalue: float
        p value
    Returns:
    -------
    str of p value
    """
    if pvalue < 0.05:
        pvalue = "<0.05"
    else:
        pvalue = f"{pvalue:.2f}"
    return pvalue


def add_lines_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add lines to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """
    y_value = ""
    if percentile == 5:
        y_value = "fifth_percentile"
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
    fig.add_shape(
        type="line",
        x0=locations["line"][color_order.index(df_row[color_column])]["x0"],
        y0=df_row[y_value],
        x1=locations["line"][color_order.index(df_row[color_column])]["x1"],
        y1=df_row[y_value],
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
        row=row,
        col=col,
    )
    return fig


def add_text_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add text to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """

    y_value = ""
    y_percent_value = ""
    y_offset = 0
    if percentile == 5:
        y_value = "fifth_percentile"
        y_percent_value = "percent_fifth_percentile"
        y_offset = -0.08
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
        y_percent_value = "percent_ninetyfifth_percentile"
        y_offset = 0.08
    fig.add_annotation(
        x=locations["text"][color_order.index(df_row[color_column])]["x"],
        y=df_row[y_value] + y_offset,
        text=f"{df_row[y_percent_value]*100:.02f}%",
        showarrow=False,
        font=dict(
            size=16,
        ),
        row=row,
        col=col,
    )
    return fig


class AveragePrecision(object):
    """
    Calculate average precision
    Parameters:
    -----------
    profile: pandas.DataFrame of profiles
    match_dict: dictionary with information about matching profiles
    reference_dict: dictionary with information about reference profiles
    n_reference: number of reference profiles
    random_baseline_ap: pandas.DataFrame with average precision of random baseline
    anti_match: boolean, if True, calculate anti-match average precision
    """

    def __init__(
        self,
        profile,
        match_dict,
        reference_dict,
        n_reference,
        random_baseline_ap,
        anti_match=False,
    ):
        self.profile = profile
        self.match_dict = match_dict
        self.reference_dict = reference_dict
        self.n_reference = n_reference
        self.random_baseline_ap = random_baseline_ap
        self.anti_match = anti_match

        self.ap = self.calculate_average_precision()
        self.map = self.calculate_mean_AP(self.ap)
        self.fp = self.calculate_fraction_positive(self.map,self.ap)

    def calculate_average_precision(self):
        """
        Calculate average precision
        Returns:
        -------
        ap_df: dataframe with average precision values
        """
        _ap_df = pd.DataFrame(
            columns=self.match_dict["matching"]
            + ["n_matches", "n_reference", "ap", "correction", "ap_corrected"]
        )
        # Filter out profiles
        #print('Matchdict',self.match_dict)
        #print(self.profile)
        #print(self.match_dict)
        #print(self.filter_profiles(self.profile, self.match_dict))
        #print('11',self.n_reference)
        
        if "filter" in self.match_dict:
            profile_matching = self.filter_profiles(self.profile, self.match_dict)
        else:
            profile_matching = self.profile.copy()
        #print('dict',self.reference_dict)
        if "filter" in self.reference_dict:
            #print('here is proffffffffing')
            profile_reference = self.filter_profiles(self.profile, self.reference_dict)
            #print('hahahahha',self.profile)
            #print('wuwuww???',self.reference_dict)
            #print('eeerrr',profile_reference)
        else:
            
            profile_reference = self.profile.copy()
            #print('wwwwwwwwa',profile_reference)

        for group_index, group in tqdm(
            profile_matching.groupby(self.match_dict["matching"])
        ):
            for index, row in group.iterrows():
                _ap_dict = {}
                profile_matching_remaining = group.drop(index)

                # Remove matches that match columns of the query
                if "non_matching" in self.match_dict:
                    profile_matching_remaining = self.remove_non_matching_profiles(
                        row, profile_matching_remaining, self.match_dict
                    )

                # Keep those reference profiles that match columns of the query
                if "matching" in self.reference_dict:
                    query_string = " and ".join(
                        [f"{_}==@row['{_}']" for _ in self.reference_dict["matching"]]
                    )
                    if not query_string == "":

                        profile_reference_remaining = profile_reference.query(
                            query_string
                        ).reset_index(drop=True)
                else:
                    profile_reference_remaining = profile_reference.copy()

                # Remove those reference profiles that do not match columns of the query
                if "non_matching" in self.reference_dict:
                    profile_reference_remaining = self.remove_non_matching_profiles(
                        row, profile_reference_remaining, self.reference_dict
                    )
                
                
                
                #print(('22',profile_reference_remaining))
                # subsample reference
                k = min(self.n_reference, len(profile_reference_remaining))
                profile_reference_remaining = profile_reference_remaining.sample(
                    k
                ).reset_index(drop=True)

                # Combine dataframes
                profile_combined = pd.concat(
                    [profile_matching_remaining, profile_reference_remaining], axis=0
                ).reset_index(drop=True)

                # Extract features
                query_perturbation_features = row[
                    ~self.profile.columns.str.startswith("Metadata")
                ]
                profile_combined_features = get_featuredata(profile_combined)

                # Compute cosine similarity
                
                #print('profile_matching_remaining',profile_matching_remaining.iloc[:,:5])
                #print('profile_reference_remaining',profile_reference_remaining.iloc[:,:5])
                
                y_true = [1] * len(profile_matching_remaining) + [0] * len(
                    profile_reference_remaining
                )

                if np.sum(y_true) == 0:
                    continue
                else:
                    y_pred = cosine_similarity(
                        query_perturbation_features.values.reshape(1, -1),
                        profile_combined_features,
                    )[0]

                    if self.anti_match:
                        y_pred = np.abs(y_pred)

                    score = average_precision_score(y_true, y_pred)

                # Correct ap using the random baseline ap

                n_matches = np.sum(y_true)
                
                n_reference = k
                
                
                #print('n_matches',n_matches)
                #print('n_reference',n_reference)
                if (
                    self.random_baseline_ap.query(
                        "n_matches == '@n_matches' and n_reference == '@n_reference'"
                    ).empty
                    == True
                    and n_matches != 0
                ):
                    self.compute_random_baseline(n_matches, n_reference)

                    correction = self.random_baseline_ap.query(
                        "n_matches == @n_matches and n_reference == @n_reference"
                    )["ap"].quantile(0.95)

                else:
                    correction = 0

                for match in self.match_dict["matching"]:
                    _ap_dict[match] = row[match]
                _ap_dict["n_matches"] = int(n_matches)
                _ap_dict["n_reference"] = int(n_reference)
                _ap_dict["ap"] = score
                _ap_dict["correction"] = correction
                _ap_dict["ap_corrected"] = score - correction
                _ap_df = pd.concat(
                    [_ap_df, pd.DataFrame(_ap_dict, index=[0])],
                    axis=0,
                    ignore_index=True,
                )

        return _ap_df

    def compute_random_baseline(self, n_matches, n_reference):
        """
        Compute the random baseline for the average precision score
        Parameters
        ----------
        n_matches: int
            Number of matches
        n_reference: int
            Number of reference profiles
        """
        if (
            self.random_baseline_ap.query(
                "n_matches == @n_matches and n_reference == @n_reference"
            ).empty
            == True
        ):
            ranked_list = [i for i in range(n_matches + n_reference)]
            truth_values = [1 for i in range(n_matches)] + [
                0 for i in range(n_reference)
            ]

            for _ in range(10000):  # number of random permutations
                random.shuffle(ranked_list)
                random.shuffle(truth_values)

                self.random_baseline_ap = pd.concat(
                    [
                        self.random_baseline_ap,
                        pd.DataFrame(
                            {
                                "ap": average_precision_score(
                                    truth_values, ranked_list
                                ),
                                "n_matches": [n_matches],
                                "n_reference": [n_reference],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    @staticmethod
    def filter_profiles(_profiles, _dict):
        """
        Filter profiles based on the filter dictionary
        Parameters
        ----------
        _profiles : pandas.DataFrame of profiles
        _dict : dictionary with filter columns
        Returns
        -------
        _profiles : pandas.DataFrame of filtered profiles
        """
        query_string = " and ".join(
            [
                " and ".join([f"{k}!={vi}" for vi in v])
                for k, v in _dict["filter"].items()
            ]
        )
        #print('qqqqqqqqqqqqqqqq',query_string)
        if not query_string == "":
            _profiles = _profiles.query(query_string).reset_index(drop=True)
         #   print('profilesPPPPPPPPPPPPPPPPPPPPPPpppp',_profiles)
        return _profiles

    @staticmethod
    def remove_non_matching_profiles(_query_profile, _profiles, _dict):
        """
        Remove profiles that match the query profile in the non_matching columns
        Parameters
        ----------
        _query_profile : pandas.Series of query profile
        _profiles : pandas.DataFrame of profiles
        _dict : dictionary with non_matching columns
        Returns
        -------
        _profiles : pandas.DataFrame of filtered profiles
        """
        for _ in _dict["non_matching"]:
            matching_col = [_query_profile[_] for i in range(len(_profiles))]
            _profiles = _profiles.loc[
                [
                    len(np.intersect1d(x[0].split("|"), x[1].split("|"))) == 0
                    for x in zip(_profiles[_], matching_col)
                ]
            ]
        return _profiles

    def calculate_mean_AP(self, _ap):
        """
        Calculate the mean average precision
        Parameters
        ----------
        _ap : pandas.DataFrame of average precision values
        Returns
        -------
        _map_df : pandas.DataFrame of mAP values gropued by matching columns
        """
        _map_df = (
            _ap.groupby(self.match_dict["matching"])
            .ap_corrected.mean()
            .reset_index()
            .rename(columns={"ap_corrected": "mAP"})
        )
        return _map_df

    @staticmethod
    def calculate_fraction_positive(_map_df,ap):
        """
        Calculate the fraction of positive matches
        Parameters
        ----------
        _map_df : pandas.DataFrame of mAP values
        Returns
        -------
        _fp : float of fraction positive
        """
        if len(_map_df)==0:
            _fp = len(_map_df.query("mAP>0")) / 1
            return _fp
        else:
        #print('AP',ap)
        #print(_map_df)
        #print(len(_map_df.query("mAP>0")))
            print(len(_map_df))
            _fp = len(_map_df.query("mAP>0")) / len(_map_df)
            print('_fp=',_fp)
        return _fp

