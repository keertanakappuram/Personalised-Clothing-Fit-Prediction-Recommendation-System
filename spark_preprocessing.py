"""
spark_preprocessing.py — PySpark Preprocessing Pipeline
=========================================================
Distributed preprocessing pipeline for the Rent The Runway dataset.
Simulates production-scale data processing using Apache Spark.

Handles:
- Raw JSON ingestion
- Height/weight parsing via UDFs
- BMI computation
- Feature engineering at scale
- User/item history aggregations
- Output to Parquet for downstream modeling

Usage:
    spark-submit spark_preprocessing.py \
        --input renttherunway_final_data.json.gz \
        --output data/processed/
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructType, StructField
)
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler as SparkStandardScaler, Bucketizer
)
from pyspark.ml import Pipeline as SparkPipeline
import re

# ─────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────
def get_spark(app_name: str = "ClothingFitPreprocessing") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ─────────────────────────────────────────────
# UDFs
# ─────────────────────────────────────────────
@F.udf(returnType=DoubleType())
def parse_height_udf(height_str):
    """Parse height string (e.g. '5\\'6\"') to total inches."""
    try:
        if not height_str:
            return None
        height_str = str(height_str).strip().replace('"', '').replace("\\'", "'")
        if "'" in height_str:
            parts = height_str.split("'")
            feet  = int(parts[0].strip())
            inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
            return float(feet * 12 + inches)
    except:
        return None

@F.udf(returnType=DoubleType())
def parse_weight_udf(weight_str):
    """Parse weight string (e.g. '140lbs') to numeric pounds."""
    try:
        if not weight_str:
            return None
        cleaned = re.sub(r'[^0-9.]', '', str(weight_str))
        return float(cleaned) if cleaned else None
    except:
        return None

@F.udf(returnType=IntegerType())
def text_mentions_small_udf(text):
    if not text: return 0
    t = str(text).lower()
    return int(any(w in t for w in ['too small', 'tight', 'snug', 'ran small', 'size up']))

@F.udf(returnType=IntegerType())
def text_mentions_large_udf(text):
    if not text: return 0
    t = str(text).lower()
    return int(any(w in t for w in ['too big', 'large', 'loose', 'baggy', 'ran large']))

@F.udf(returnType=IntegerType())
def text_mentions_perfect_udf(text):
    if not text: return 0
    t = str(text).lower()
    return int(any(w in t for w in ['perfect', 'true to size', 'fit perfectly', 'just right']))

@F.udf(returnType=DoubleType())
def size_to_numeric_udf(size_str):
    size_map = {'xxs': 0, 'xs': 1, 's': 2, 'sm': 2, 'm': 3, 'md': 3,
                'l': 4, 'lg': 4, 'xl': 5, 'xxl': 6, '1x': 5, '2x': 6, '3x': 7}
    try:
        if not size_str: return None
        return float(size_map.get(str(size_str).lower().strip(), None))
    except:
        return None

# ─────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────
def load_data(spark: SparkSession, input_path: str):
    """Load raw JSON data into Spark DataFrame."""
    print(f"Loading data from: {input_path}")
    df = spark.read.json(input_path)
    print(f"Records loaded: {df.count():,}")
    print(f"Schema: {df.printSchema()}")
    return df

def clean_data(df):
    """Clean and parse raw fields."""
    print("Cleaning data...")

    df = df.withColumn("height_inches", parse_height_udf(F.col("height")))
    df = df.withColumn("weight_lbs",    parse_weight_udf(F.col("weight")))
    df = df.withColumn("bmi",
        F.when(
            (F.col("height_inches") > 0) & (F.col("weight_lbs") > 0),
            703 * F.col("weight_lbs") / (F.col("height_inches") * F.col("height_inches"))
        ).otherwise(None)
    )
    df = df.withColumn("age", F.col("age").cast(DoubleType()))
    df = df.withColumn("age",
        F.when((F.col("age") >= 18) & (F.col("age") <= 80), F.col("age")).otherwise(None)
    )
    df = df.withColumn("fit", F.lower(F.trim(F.col("fit"))))
    df = df.filter(F.col("fit").isin("small", "fit", "large"))
    df = df.filter((F.col("bmi") >= 15) & (F.col("bmi") <= 50))
    df = df.withColumn("size_numeric", size_to_numeric_udf(F.col("size")))
    df = df.withColumn("rating_numeric", F.col("rating").cast(DoubleType()))

    # Fill nulls with medians (approximation via Spark)
    median_vals = df.approxQuantile(
        ["height_inches", "weight_lbs", "bmi", "age", "size_numeric", "rating_numeric"],
        [0.5], 0.01
    )
    fill_map = {
        "height_inches": median_vals[0][0],
        "weight_lbs":    median_vals[1][0],
        "bmi":           median_vals[2][0],
        "age":           median_vals[3][0],
        "size_numeric":  median_vals[4][0],
        "rating_numeric":median_vals[5][0]
    }
    df = df.fillna(fill_map)
    df = df.fillna({"body type": "unknown", "rented for": "unknown", "review_text": ""})

    print(f"Records after cleaning: {df.count():,}")
    return df

def engineer_features(df):
    """Feature engineering at scale."""
    print("Engineering features...")

    # BMI buckets
    bmi_splits = [0, 18.5, 25.0, 30.0, 35.0, float('inf')]
    bucketizer = Bucketizer(splits=bmi_splits, inputCol="bmi", outputCol="bmi_bucket")
    df = bucketizer.transform(df)

    # Size relative to BMI peers
    bmi_peer_avg = df.groupBy("bmi_bucket").agg(
        F.avg("size_numeric").alias("peer_avg_size")
    )
    df = df.join(bmi_peer_avg, on="bmi_bucket", how="left")
    df = df.withColumn("size_relative_to_peers",
                       F.col("size_numeric") - F.col("peer_avg_size"))

    # Text signals
    df = df.withColumn("text_small",   text_mentions_small_udf(F.col("review_text")))
    df = df.withColumn("text_large",   text_mentions_large_udf(F.col("review_text")))
    df = df.withColumn("text_perfect", text_mentions_perfect_udf(F.col("review_text")))

    # Rating flag
    df = df.withColumn("is_high_rating",
                       F.when(F.col("rating_numeric") >= 9, 1).otherwise(0))

    # Interaction features
    df = df.withColumn("bmi_size_interaction",
                       F.col("bmi") * F.col("size_numeric"))
    df = df.withColumn("size_per_bmi",
                       F.col("size_numeric") / (F.col("bmi") + 1e-9))

    return df

def add_history_features(df):
    """
    Compute user and item history features.
    Uses only training data (fit != null) to avoid leakage.
    In production this would be a separate pre-computed table.
    """
    print("Computing user/item history features...")

    # User history
    user_history = df.groupBy("user_id").agg(
        F.avg(F.when(F.col("fit") == "small", 1).otherwise(0)).alias("user_small_rate"),
        F.avg(F.when(F.col("fit") == "large", 1).otherwise(0)).alias("user_large_rate"),
        F.avg(F.when(F.col("fit") == "fit",   1).otherwise(0)).alias("user_fit_rate"),
        F.avg("size_numeric").alias("user_avg_size"),
        F.count("fit").alias("user_total_rentals")
    )

    # Item history
    item_history = df.groupBy("item_id").agg(
        F.avg(F.when(F.col("fit") == "small", 1).otherwise(0)).alias("item_small_rate"),
        F.avg(F.when(F.col("fit") == "large", 1).otherwise(0)).alias("item_large_rate"),
        F.count("fit").alias("item_total_rentals")
    )
    item_history = item_history.withColumn(
        "item_runs_small", F.when(F.col("item_small_rate") > 0.20, 1).otherwise(0)
    ).withColumn(
        "item_runs_large", F.when(F.col("item_large_rate") > 0.20, 1).otherwise(0)
    )

    df = df.join(user_history, on="user_id", how="left")
    df = df.join(item_history, on="item_id", how="left")

    df = df.withColumn("is_cold_user", F.when(F.col("user_small_rate").isNull(), 1).otherwise(0))
    df = df.withColumn("is_cold_item", F.when(F.col("item_small_rate").isNull(), 1).otherwise(0))

    history_cols = ["user_small_rate", "user_large_rate", "user_fit_rate",
                    "user_avg_size", "user_total_rentals",
                    "item_small_rate", "item_large_rate", "item_total_rentals",
                    "item_runs_small", "item_runs_large"]
    df = df.fillna({col: 0.0 for col in history_cols})

    return df

def encode_target(df):
    """Encode fit label as numeric index."""
    label_indexer = StringIndexer(
        inputCol="fit",
        outputCol="fit_encoded",
        stringOrderType="alphabetAsc"  # fit=0, large=1, small=2
    )
    df = label_indexer.fit(df).transform(df)
    return df

def save_output(df, output_path: str):
    """Save processed data to Parquet."""
    output_cols = [
        "user_id", "item_id", "fit", "fit_encoded",
        "age", "bmi", "height_inches", "weight_lbs", "size_numeric",
        "size_relative_to_peers", "bmi_size_interaction", "size_per_bmi",
        "user_small_rate", "user_large_rate", "user_fit_rate",
        "user_avg_size", "user_total_rentals",
        "item_small_rate", "item_large_rate", "item_total_rentals",
        "item_runs_small", "item_runs_large",
        "rating_numeric", "is_high_rating",
        "is_cold_user", "is_cold_item",
        "text_small", "text_large", "text_perfect",
        "review_text"
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    print(f"Saving {df.count():,} records to {output_path}...")
    df.select(output_cols).write.mode("overwrite").parquet(output_path)
    print(f"Saved to {output_path} ✅")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(input_path: str, output_path: str):
    print("=" * 60)
    print("  Clothing Fit — PySpark Preprocessing Pipeline")
    print("=" * 60)

    spark = get_spark()

    df = load_data(spark, input_path)
    df = clean_data(df)
    df = engineer_features(df)
    df = add_history_features(df)
    df = encode_target(df)

    # Summary stats
    print("\nDataset summary:")
    df.groupBy("fit").count().orderBy("fit").show()
    df.select(
        F.count("*").alias("total_records"),
        F.countDistinct("user_id").alias("unique_users"),
        F.countDistinct("item_id").alias("unique_items"),
        F.avg("bmi").alias("avg_bmi"),
        F.avg("user_total_rentals").alias("avg_user_rentals")
    ).show()

    save_output(df, output_path)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="renttherunway_final_data.json.gz")
    parser.add_argument("--output", type=str, default="data/processed/")
    args = parser.parse_args()
    main(args.input, args.output)
