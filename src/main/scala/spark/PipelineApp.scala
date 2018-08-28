import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object PipelineApp {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("Pipeline Application").getOrCreate()
        val training = spark.createDataFrame(
            Seq(
                (0L, "a b c d e Spark", 1.0),
                (1L, "b d", 0.0),
                (2L, "Spark f g h", 1.0),
                (3L, "Hadoop MapReduce", 0.0)
            )
        ).toDF("id", "text", "label")
        val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
        val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
        val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001)
        val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
        val model = pipeline.fit(training)
        val test = spark.createDataFrame(
            Seq(
                (4L, "Spark i j k"),
                (5L, "l m n"),
                (6L, "Spark Hadoop Spark"),
                (7L, "Apache Hadoop")
            )
        ).toDF("id", "text")
        model
            .transform(test)
            .select("id", "text", "probability", "prediction")
            .collect()
            .foreach {
                case Row(id: Long, text: String, prob: Vector, prediction: Double) => 
                    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
            }
        spark.stop()
    }
}