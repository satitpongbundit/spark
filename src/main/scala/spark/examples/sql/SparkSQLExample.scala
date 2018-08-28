import org.apache.spark.sql.SparkSession

case class Person(name: String, age: Long)

object SparkSQLExample {
    def main(args: Array[String]) {
        val spark = SparkSession
            .builder()
            .appName("Spark SQL basic example")
            .config("spark.some.config.option", "some-value")
            .getOrCreate()

        // For implicit conversions like converting RDDs to DataFrames
        import spark.implicits._

        val df = spark.read.json("src/main/resources/people.json")

        // Displays the content of the DataFrame to stdout
        df.show()

        // Print the schema in a tree format
        df.printSchema()

        // Select only the "name" column
        df.select("name").show()

        // Select everybody, but increment the age by 1
        df.select($"name", $"age" + 1).show()

        // Select people older than 21
        df.filter($"age" > 21).show()

        // Count people by age
        df.groupBy("age").count().show()

        // Register the DataFrame as a SQL temporary view
        df.createOrReplaceTempView("people")

        val sqlDF = spark.sql("SELECT * FROM people")
        sqlDF.show()

        // Register the DataFrame as a global temporary view
        df.createGlobalTempView("people")

        // Global temporary view is tied to a system preserved database `global_temp`
        spark.sql("SELECT * FROM global_temp.people").show()

        // Global temporary view is cross-session
        spark.newSession().sql("SELECT * FROM global_temp.people").show()

        // Encoders are created for case classes
        val caseClassDS = Seq(Person("Andy", 32)).toDS()
        caseClassDS.show()

        // Encoders for most common types are automatically provided by importing spark.implicits._
        val primitiveDS = Seq(1, 2, 3).toDS()
        primitiveDS.map(_ + 1).collect()

        // DataFrames can be converted to a Dataset by providing a class. Mapping will be done by name
        val path = "src/main/resources/people.json"
        val peopleDS = spark.read.json(path).as[Person]
        peopleDS.show()

        // Create an RDD of Person objects from a text file, convert it to a Dataframe
        val peopleDF = spark.sparkContext
            .textFile("src/main/resources/people.txt")
            .map(_.split(","))
            .map(attributes => Person(attributes(0), attributes(1).trim.toInt))
            .toDF()
        
        // Register the DataFrame as a temporary view
        peopleDF.createOrReplaceTempView("people")

        // SQL statements can be run by using the SQL methods provided by Spark
        val teenagersDF = spark.sql("SELECT name, age FROM people WHERE age BETWEEN 13 AND 19")

        // The columns of a row in the result can be accessed by field index
        teenagersDF.map(teenager => "Name: " + teenager(0)).show()

        // or by field name
        teenagersDF.map(teenager => "Name: " + teenager.getAs[String]("name")).show()

        // No pre-defined encoders for Dataset[Map[K,V]], define explicitly
        implicit val mapEncoder = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]

        // row.getValuesMap[T] retrieves multiple columns at once into a Map[Sring, T]
        teenagersDF.map(teenager => teenager.getValuesMap[Any](List("name", "age"))).collect()

        spark.close()
    }
}