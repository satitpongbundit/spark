import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object KMeansExample {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("K-means Example").getOrCreate()

        // loads data
        val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

        // trains a k-means model
        val kmeans = new KMeans().setK(2).setSeed(1L)
        val model = kmeans.fit(dataset)

        // make predictions
        val predictions = model.transform(dataset)

        // evaluate clustering by computing Silhouette score
        val evaluator = new ClusteringEvaluator()
        val silhouette = evaluator.evaluate(predictions)
        println(s"Silhouette with squared euclidean distance = $silhouette")

        // shows the result
        println("Cluster Centers:")
        model.clusterCenters.foreach(println)

        spark.close()
    }
}