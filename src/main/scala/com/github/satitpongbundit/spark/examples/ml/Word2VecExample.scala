import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object Word2VecExample {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("Word2Vec Example").getOrCreate()
        import spark.implicits._
        val documentDF = spark.createDataFrame(
            Seq(
                "Hi, I heard about Spark.".split(" "),
                "I wish Java could use case classes.".split(" "),
                "Logistic Regression models are neat.".split(" ")
            ).map(Tuple1.apply)
        ).toDF("text")
        val word2vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
        val model = word2vec.fit(documentDF)
        val result = model.transform(documentDF)
        result.collect().foreach {
            case Row(text: Seq[_], features: Vector) =>
                println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")
        }
        spark.close()
    }
}