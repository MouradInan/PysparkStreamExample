package sparkml_training

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql. SparkSession
import org.apache.spark.sql.functions.explode

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

object TP2 {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    // Create Spark Context
    val spark = SparkSession
      .builder()
      .appName("TP1")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val ratings = spark.read
      .option("delimiter", "\t")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("./src/data/u.data")

    // Train-test split
    val Array(train,test) = ratings.randomSplit(Array(0.8,0.2))


    // Create ALS model and fit it
    val als = new ALS()
      .setItemCol("item_id")
      .setUserCol("user_id")
      .setRatingCol("rating")

    val model = als.fit(train).setColdStartStrategy("drop")

    // Prediction
    val results = model.transform(test)


    // RMSE Computation
    val regEval = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = regEval.evaluate(results)

    println(s"The computed RMSE is $rmse")

    val usersRecs = model.recommendForAllUsers(10)
    val moviesRecs = model.recommendForAllItems(10)


    usersRecs.show()
    moviesRecs.show()

    model.userFactors.show()


    val user = model.userFactors.where($"id" === 471).first()(1).asInstanceOf[Seq[Float]].zipWithIndex
    val movie = model.itemFactors.where($"id" === 1233).first()(1).asInstanceOf[Seq[Float]].zipWithIndex

    println(user)
    println(movie)

    val product =  user.map{ case (v,i) => v * movie(i)._1 }.sum

    println(s"Score with matrix multiplication : $product")

    usersRecs
      .withColumn("r",explode($"recommendations"))
      .where($"user_id" === 471 && $"r.item_id" === 1233)
      .show()

  }

}
