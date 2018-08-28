import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.SparkSession

object ChiSquareTestApp {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("Chi Square Test Application").getOrCreate()
        import spark.implicits._
        val data = Seq(
            (0.0, Vectors.dense(0.5, 10.0)),
            (0.0, Vectors.dense(1.5, 20.0)),
            (1.0, Vectors.dense(1.5, 30.0)),
            (0.0, Vectors.dense(3.5, 30.0)),
            (0.0, Vectors.dense(3.5, 40.0)),
            (1.0, Vectors.dense(3.5, 40.0))
        )
        val df = data.toDF("label", "features")
        val chi = ChiSquareTest.test(df, "features", "label").head
        println(s"pValues = ${chi.getAs[Vector](0)}")
        println(s"DegreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
        println(s"Statistics ${chi.getAs[Vector](2)}")
        spark.close()
    }
}