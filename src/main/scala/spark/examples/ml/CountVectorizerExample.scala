import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

object CountVectorizerExample {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("CountVectorizer Example").getOrCreate()
        val df = spark.createDataFrame(
            Seq(
                (0.0, Array("a", "b", "c")),
                (1.0, Array("a", "b", "b", "c", "a"))
            )
        ).toDF("id", "words")

        //  fit a CountVectorizerModel from the corpus
        val cvModel: CountVectorizerModel = new CountVectorizer()
            .setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2).fit(df)

        // alternatively, define CountVectorizerModel with a-priory vocabulary
        val cvm = new CountVectorizerModel(Array("a", "b", "c")).setInputCol("words").setOutputCol("features")

        cvModel.transform(df).show(false)
        
        spark.close()
    }
}