
#import necessary libraries
from pathlib import Path

import numpy as np
import pandas as pd

#define the function to return the directory containing the historical CSV files
def get_historical_data_dir() -> Path:
    """Return the directory containing the historical CSV files."""
    return Path(__file__).resolve().parent / "energyanalystcodingexercise" / "historicalPriceData"


#TASK 1
#define the function to count rows in each CSV file
def count_rows_in_each_csv(data_dir: Path) -> dict[str, int]:
    """Count the number of rows in each CSV file and return a dictionary of counts."""
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    row_counts = {}
    for file in csv_files:
        df = pd.read_csv(file)
        row_counts[file.name] = len(df)

    return row_counts

#define the function to load and combine historical data from CSV files
def load_historical_data() -> pd.DataFrame:
    """Read all historical price CSV files and combine them into one dataframe."""
    data_dir = get_historical_data_dir()
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(frames, ignore_index=True)

    if "Date" in combined_df.columns:
        combined_df["Date"] = pd.to_datetime(combined_df["Date"])

    return combined_df

#define the function to validate that the combined data matches the total rows in individual CSV files
def validate_combined_data() -> pd.DataFrame:
    """Count rows in each CSV, then verify the concatenated dataframe has the same total."""
    data_dir = get_historical_data_dir()
    row_counts = count_rows_in_each_csv(data_dir)

    combined_df = load_historical_data()
    combined_row_count = len(combined_df)
    total_rows_in_individual_csvs = sum(row_counts.values())

    print("Rows in each CSV file:")
    for file_name, count in row_counts.items():
        print(f"  {file_name}: {count}")

    print(f"\nTotal rows across all CSV files: {total_rows_in_individual_csvs}")
    print(f"Rows in combined dataframe: {combined_row_count}")

    if total_rows_in_individual_csvs != combined_row_count:
        raise ValueError(
            "Row count mismatch: the combined dataframe does not contain all rows from the CSV files."
        )

    print("Validation passed: combined dataframe rows match the total rows in all CSV files.")
    return combined_df


if __name__ == "__main__":
    combined_df = validate_combined_data()
    print("\nFirst few rows of combined data:")
    print(combined_df.head())


#call the function to load historical data, validate that it was concatenated properly, and store it in a variable
combined_df = validate_combined_data()

print(combined_df.head())  # Display the first few rows of the combined dataframe

#TASK 2
#Compute monthly average prices for each settlement point across the full historical dataset.
#This includes both hubs (HB_) and load zones (LZ_) and keeps all prices, including negative values.
def compute_monthly_average_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return the monthly average price for each settlement point in the historical dataset."""
    if "Date" not in df.columns or "SettlementPoint" not in df.columns or "Price" not in df.columns:
        raise ValueError("The dataframe must contain Date, SettlementPoint, and Price columns.")

    monthly_avg_df = df.copy()
    monthly_avg_df["Date"] = pd.to_datetime(monthly_avg_df["Date"])
    monthly_avg_df["Year"] = monthly_avg_df["Date"].dt.year
    monthly_avg_df["Month"] = monthly_avg_df["Date"].dt.month

    monthly_avg_df = (
        monthly_avg_df.groupby(["SettlementPoint", "Year", "Month"], as_index=False)["Price"]
        .mean()
        .rename(columns={"Price": "AveragePrice"})
        .sort_values(["Year", "Month", "SettlementPoint"])
        .reset_index(drop=True)
    )

    return monthly_avg_df


monthly_avg_prices = compute_monthly_average_prices(combined_df)
print(monthly_avg_prices.head())
print(f"Number of rows in monthly averages: {len(monthly_avg_prices)}")

#Check whether each settlement point has all 48 year-month records from Jan 2016 through Dec 2019.
def check_month_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Print the number of unique year-month records per settlement point and identify missing months."""
    if "Date" not in df.columns or "SettlementPoint" not in df.columns:
        raise ValueError("The dataframe must contain Date and SettlementPoint columns.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    expected_year_months = set(
        (year, month)
        for year in range(2016, 2020)
        for month in range(1, 13)
    )

    coverage = (
        df.groupby(["SettlementPoint", "Year", "Month"])
        .size()
        .reset_index()
        .rename(columns={0: "Count"})
    )

    coverage_by_point = (
        coverage.groupby("SettlementPoint")[["Year", "Month"]]
        .size()
        .reset_index(name="YearMonthCount")
    )

    print("Year-month coverage by settlement point:")
    print(coverage_by_point.sort_values("YearMonthCount"))

    missing = []
    for settlement_point, group in df.groupby("SettlementPoint"):
        present = set(zip(group["Year"], group["Month"]))
        missing_months = sorted(expected_year_months - present)
        if missing_months:
            missing.append({
                "SettlementPoint": settlement_point,
                "MissingYearMonthPairs": missing_months,
                "MissingCount": len(missing_months),
            })

    if missing:
        print("\nSettlement points missing month coverage:")
        for item in missing:
            print(
                f"  {item['SettlementPoint']}: "
                f"{item['MissingCount']} missing months -> {item['MissingYearMonthPairs']}"
            )
    else:
        print("\nAll settlement points have complete 2016-2019 monthly coverage.")

    return coverage_by_point


check_month_coverage(combined_df)

#TASK 3
#Write the computed monthly average prices to a CSV named AveragePriceByMonth.csv
monthly_avg_prices.to_csv("AveragePriceByMonth.csv", index=False)

#TASK 4
#Compute annual hourly price volatility for each settlement hub (HB_ prefix) in the historical dataset.
#Use log returns on positive prices only, and exclude all load zones (LZ_).
def compute_hourly_price_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Return annual volatility for each HUB settlement point using log returns on positive prices."""
    required_cols = {"Date", "SettlementPoint", "Price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("The dataframe must contain Date, SettlementPoint, and Price columns.")

    hub_df = df.copy()
    hub_df["Date"] = pd.to_datetime(hub_df["Date"])
    hub_df = hub_df[hub_df["SettlementPoint"].str.startswith("HB_")].copy()

    # Keep only positive prices for log-return calculations.
    hub_df = hub_df[hub_df["Price"] > 0].copy()
    if hub_df.empty:
        raise ValueError("No positive prices remain for HB_ settlement points after filtering.")

    hub_df = hub_df.sort_values(["SettlementPoint", "Date"]).reset_index(drop=True)
    hub_df["Year"] = hub_df["Date"].dt.year

    # Compute log returns: ln(P_t / P_{t-1})
    hub_df["LogReturn"] = (
        hub_df.groupby("SettlementPoint")["Price"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )

    # Remove the first observation for each settlement point because it has no prior return.
    hub_df = hub_df.dropna(subset=["LogReturn"]).copy()

    volatility_df = (
        hub_df.groupby(["SettlementPoint", "Year"], as_index=False)["LogReturn"]
        .std(ddof=1)
        .rename(columns={"LogReturn": "HourlyVolatility"})
        .sort_values(["Year", "SettlementPoint"])
        .reset_index(drop=True)
    )

    return volatility_df


hub_volatility = compute_hourly_price_volatility(combined_df)
print(hub_volatility.head())
print(f"Number of rows in volatility output: {len(hub_volatility)}")

#TASK 5
#Write the computed hourly volatilities to a CSV named HourlyVolatilityByYear.csv.
#The file must contain exactly three columns: SettlementPoint, Year, HourlyVolatility.
def write_hourly_volatility_to_csv(df: pd.DataFrame, output_path: str = "HourlyVolatilityByYear.csv") -> None:
    """Compute annual HUB volatility and write it to a CSV with the required column names."""
    volatility_df = compute_hourly_price_volatility(df)
    volatility_df = volatility_df[["SettlementPoint", "Year", "HourlyVolatility"]]
    volatility_df.to_csv(output_path, index=False)
    print(f"Saved hourly volatility to {output_path}")
    print(volatility_df.head())


write_hourly_volatility_to_csv(combined_df)

#TASK 6
#Determine which settlement hub had the highest hourly volatility for each historical year.
def write_max_volatility_by_year(df: pd.DataFrame, output_path: str = "MaxVolatilityByYear.csv") -> None:
    """Select the settlement hub with the highest volatility in each year and write to a CSV."""
    volatility_df = compute_hourly_price_volatility(df)

    max_volatility_df = (
        volatility_df.sort_values(["Year", "HourlyVolatility", "SettlementPoint"], ascending=[True, False, True])
        .drop_duplicates(subset=["Year"], keep="first")
        .reset_index(drop=True)
    )

    max_volatility_df = max_volatility_df[["SettlementPoint", "Year", "HourlyVolatility"]]
    max_volatility_df.to_csv(output_path, index=False)

    print(f"Saved maximum annual volatility by hub to {output_path}")
    print(max_volatility_df)


write_max_volatility_by_year(combined_df)

#TASK 7
#Translate the hourly historical data into the cQuant model input format.
#Each settlement point gets its own CSV file with one daily row per date and 24 hourly columns.
def format_historical_spot_history(df: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
    """Write one cQuant-ready CSV per settlement point using the supplemental example format."""
    required_cols = {"Date", "SettlementPoint", "Price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("The dataframe must contain Date, SettlementPoint, and Price columns.")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "formattedSpotHistory"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_df = df.copy()
    formatted_df["Date"] = pd.to_datetime(formatted_df["Date"])
    formatted_df["DateOnly"] = formatted_df["Date"].dt.strftime("%Y-%m-%d")
    formatted_df["Hour"] = formatted_df["Date"].dt.hour + 1

    output_files = []
    for settlement_point, group in formatted_df.groupby("SettlementPoint"):
        pivoted = (
            group.pivot(index="DateOnly", columns="Hour", values="Price")
            .reindex(columns=range(1, 25), fill_value=None)
            .reset_index()
            .rename(columns={"DateOnly": "Date"})
        )

        pivoted.insert(0, "Variable", settlement_point)
        pivoted = pivoted.rename(columns={hour: f"X{hour}" for hour in range(1, 25)})
        pivoted["Date"] = pd.to_datetime(pivoted["Date"]).dt.strftime("%Y-%m-%d")

        output_file = output_dir / f"spot_{settlement_point}.csv"
        pivoted.to_csv(output_file, index=False)
        output_files.append(output_file)

    print(f"Saved {len(output_files)} settlement-point files to {output_dir}")
    return output_files


formatted_spot_history_files = format_historical_spot_history(combined_df)
print(formatted_spot_history_files[:3])

#BONUS MEAN PLOTS
import matplotlib.pyplot as plt


def plot_monthly_average_prices_by_type(monthly_avg_df: pd.DataFrame) -> None:
    """Create two line plots of monthly average prices: one for hubs and one for load zones."""
    if {"SettlementPoint", "Year", "Month", "AveragePrice"}.difference(monthly_avg_df.columns):
        raise ValueError("The dataframe must contain SettlementPoint, Year, Month, and AveragePrice columns.")

    monthly_plot_df = monthly_avg_df.copy()
    monthly_plot_df["Date"] = pd.to_datetime(
        monthly_plot_df["Year"].astype(str) + "-" + monthly_plot_df["Month"].astype(str) + "-01"
    )

    hub_df = monthly_plot_df[monthly_plot_df["SettlementPoint"].str.startswith("HB_")].copy()
    load_zone_df = monthly_plot_df[monthly_plot_df["SettlementPoint"].str.startswith("LZ_")].copy()

    for settlement_type, data in [("HB_", hub_df), ("LZ_", load_zone_df)]:
        if data.empty:
            continue

        plt.figure(figsize=(14, 8))
        for settlement_point, group in data.groupby("SettlementPoint"):
            plt.plot(
                group["Date"],
                group["AveragePrice"],
                label=settlement_point,
                linewidth=2,
            )

        plt.title(f"Monthly Average Prices for { 'Settlement Hubs' if settlement_type == 'HB_' else 'Load Zones' }")
        plt.xlabel("Date")
        plt.ylabel("Average Price")
        plt.xticks(rotation=45)
        plt.legend(title="Settlement Point", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        output_filename = (
            "SettlementHubAveragePriceByMonth.png"
            if settlement_type == "HB_"
            else "LoadZoneAveragePriceByMonth.png"
        )
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved plot: {output_filename}")


plot_monthly_average_prices_by_type(monthly_avg_prices)


#BONUS VOLATILITY PLOTS

def plot_hub_volatility_by_year(volatility_df: pd.DataFrame) -> None:
    """Create and save volatility plots for settlement hubs by year."""
    if {"SettlementPoint", "Year", "HourlyVolatility"}.difference(volatility_df.columns):
        raise ValueError("The dataframe must contain SettlementPoint, Year, and HourlyVolatility columns.")

    output_dir = Path(__file__).resolve().parent

    grouped_df = volatility_df.copy()
    grouped_df["Year"] = grouped_df["Year"].astype(int)

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    for settlement_point, group in grouped_df.groupby("SettlementPoint"):
        plt.plot(group["Year"], group["HourlyVolatility"], marker="o", linewidth=2, label=settlement_point)

    plt.title("Annual Hourly Volatility by Settlement Hub")
    plt.xlabel("Year")
    plt.ylabel("Hourly Volatility")
    plt.xticks(sorted(grouped_df["Year"].unique()))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Settlement Point", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    volatility_trend_path = output_dir / "HubVolatilityByYear.png"
    plt.savefig(volatility_trend_path)
    plt.close()
    print(f"Saved plot: {volatility_trend_path.name}")

    plt.figure(figsize=(12, 8))
    pivot_df = grouped_df.pivot(index="Year", columns="SettlementPoint", values="HourlyVolatility")
    pivot_df.plot(kind="bar", figsize=(12, 8), alpha=0.9)
    plt.title("Settlement Hub Volatility by Year")
    plt.xlabel("Year")
    plt.ylabel("Hourly Volatility")
    plt.xticks(rotation=0)
    plt.legend(title="Settlement Point")
    plt.tight_layout()
    volatility_bar_path = output_dir / "HubVolatilityComparisonByYear.png"
    plt.savefig(volatility_bar_path)
    plt.close()
    print(f"Saved plot: {volatility_bar_path.name}")


plot_hub_volatility_by_year(hub_volatility)


#BONUS HOURLY SHAPE PROFILE COMPUTATION

def compute_hourly_shape_profiles(df: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
    """Compute normalized hourly shape profiles by month-of-year and day-of-week for each settlement point."""
    required_cols = {"Date", "SettlementPoint", "Price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("The dataframe must contain Date, SettlementPoint, and Price columns.")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "hourlyShapeProfiles"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    profile_df = df.copy()
    profile_df["Date"] = pd.to_datetime(profile_df["Date"])
    profile_df["Month"] = profile_df["Date"].dt.month
    profile_df["DayOfWeek"] = profile_df["Date"].dt.dayofweek
    profile_df["Hour"] = profile_df["Date"].dt.hour

    output_files = []

    for settlement_point, group in profile_df.groupby("SettlementPoint"):
        hourly_profile = (
            group.groupby(["Month", "DayOfWeek", "Hour"], as_index=False)["Price"]
            .mean()
            .rename(columns={"Price": "AveragePrice"})
            .sort_values(["Month", "DayOfWeek", "Hour"])
            .reset_index(drop=True)
        )

        hourly_profile["ShapeProfile"] = (
            hourly_profile.groupby(["Month", "DayOfWeek"])["AveragePrice"]
            .transform(lambda x: x / x.mean())
        )

        profile_matrix = hourly_profile.pivot_table(
            index=["Month", "DayOfWeek"],
            columns="Hour",
            values="ShapeProfile",
            aggfunc="first",
        ).reindex(columns=range(24), fill_value=np.nan)

        profile_matrix = profile_matrix.rename(columns=lambda h: f"H{h+1}")
        profile_matrix = profile_matrix.reset_index()
        profile_matrix.insert(0, "SettlementPoint", settlement_point)

        output_file = output_dir / f"profile_{settlement_point}.csv"
        profile_matrix.to_csv(output_file, index=False)
        output_files.append(output_file)

    print(f"Saved {len(output_files)} settlement-point shape profile files to {output_dir}")
    return output_files


shape_profile_files = compute_hourly_shape_profiles(combined_df)
print(shape_profile_files[:3])


