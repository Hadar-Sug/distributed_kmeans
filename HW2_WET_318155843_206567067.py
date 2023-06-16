# All of your imports should go here
# You cannot use any premade k-means nor import sklearn...

def kmeans_fit(data: pyspark.sql.DataFrame,
               init: pyspark.sql.DataFrame,
               k: int = 4,
               max_iter: int = 10):
    # imports
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    import pyspark.sql.functions as F
    spark = SparkSession.builder.getOrCreate()

    def check_convergence(prev_centroids, centroids):
        for i in range(len(centroids)):
            if list(prev_centroids[i]) != list(centroids[i]):
                return False
        return True

    # initialization:
    columns = data.columns
    converged = False
    data_points = data.select("*").persist()
    # add a column with number 2 to speed things up late
    data_points = data_points.withColumn("pow", F.lit(2))
    # create a list of centroids
    new_centroids = [list(centroid) for centroid in init.collect()]
    # print(centroids)
    for iteration in range(max_iter):
        if converged:
            break
        else:
            prev_centroids = new_centroids
            distances = []
            for i, centroid in enumerate(prev_centroids):
                distances.append(f'dist_from_c{i}')
                # for each centroid calculate sigma{(x_j-c_j)^2} by creating (x_j-c_j)^2 column for each j
                for j in range(len(centroid)):
                    data_points = data_points.withColumn(f"x_{j}_minus_c{j}_pow",
                                                         F.expr(f"{columns[j]} - {centroid[j]}")) \
                        .withColumn(f"x_{j}_minus_c{j}_pow", F.pow(f"x_{j}_minus_c{j}_pow", "pow"))
                # now sum (x_j-c_j)^2 columns to get euclidian distance from centroid i
                # no need for sqrt since its monotonicly increasing and distances are strictly positive
                data_points = data_points.withColumn(f'dist_from_c{i}', sum([F.col(f"x_{k}_minus_c{k}_pow") for k in
                                                                             range(len(centroid))]))

            # create a column with the assigned centroid for each point
            cond = F.expr(
                "CASE " + " ".join([f"WHEN {c} = minimum THEN '{i}'" for i, c in enumerate(distances)]) + " END")
            data_points = data_points.withColumn('minimum', F.least(*distances)).withColumn("centroid_id", cond)

            # compute new centroids by averaging the points per centroid
            new_centroids = data_points.withColumn("centroid_id", F.col("centroid_id").cast("int")) \
                .groupBy("centroid_id").avg(*columns).orderBy("centroid_id")
            #
            new_centroids = [list(centroid[1:]) for centroid in new_centroids.collect()]
            # check convergence
            if iteration > 1:
                converged = check_convergence(prev_centroids, new_centroids)

    centroids = spark.createDataFrame(new_centroids)
    vecAssembler = VectorAssembler(inputCols=centroids.columns, outputCol="centroids")
    return vecAssembler.transform(centroids).select("centroids")