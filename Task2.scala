import java.io.{File, FileWriter, PrintWriter}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import scala.collection.mutable.{ListBuffer, Map, Set}


object Task2 {

  case class Cluster(id: Int, size: Int, error: Double, terms: List[String])

  case class Type(algorithm: String, WSSE: Double, clusters: List[Cluster])

  var clusterInfo: ListBuffer[Cluster] = ListBuffer()


  def euclideanDistance(mean: Array[Double], point: Array[Double]): Double = {

    var distance = 0.0
    val n = mean.size
    for (i <- 0 until n) {
      distance += Math.pow(Math.abs(mean(i) - point(i)), 2)
    }
    val res = distance
    res
  }


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("info").setLevel(Level.OFF)

    val start = System.currentTimeMillis()
    val input_path = args(0)
    val algorithm = args(1)
    val numClusters = args(2).toInt
    val numIterations = args(3).toInt

//    val input_path = "/Users/shivanishukla/Desktop/INF-553/DM-Assignment4/INF553_Assignment4/Data/yelp_reviews_clustering_small.txt"
//    val algorithm = "B"
//    val numClusters = 8
//    val numIterations = 20

    val conf = new SparkConf().setMaster("local[*]").setAppName("Clustering")
    val sc = new SparkContext(conf)

    // Load documents (one per line).
    val documents: RDD[Seq[String]] = sc.textFile(input_path)
      .map(_.split(" ").toSeq)
    val hashingTF = new HashingTF(1 << 100)
    val tf: RDD[Vector] = hashingTF.transform(documents)
    val idf = new IDF().fit(tf)
    val parsedData: RDD[Vector] = idf.transform(tf)

    if (algorithm.equals("K")) {

      val output_path = "Shivani_Shukla_Cluster_small_K_8_20.json"

      // Cluster the data into two classes using KMeans
      val clusters = KMeans.train(parsedData, numClusters, numIterations, "k-means||", 42)

      // Evaluate clustering by computing Within Set Sum of Squared Errors
      val WSSSE = clusters.computeCost(parsedData)

      val centroids = clusters.clusterCenters

      val errors_map = documents.mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[(Int, Double)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = clusters.predict(tfidf)

          val newerr = euclideanDistance(centroids(cluster_id).toArray, tfidf.toArray)
          result.append((cluster_id, newerr))

          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).sortByKey().collect()

      val size_map = documents.mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[(Int, Int)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = clusters.predict(tfidf)


          result.append((cluster_id, 1))

          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).sortByKey().collect()



      val cluster_words = documents.repartition((1)).mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[((String, Int), Int)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = clusters.predict(tfidf)


          for (word <- document) {
            result.append(((word, cluster_id), 1))
          }
          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).map(row => (row._1._2, (row._1._1, row._2))).groupByKey().sortByKey().collect()



      val topTenWords: ListBuffer[ListBuffer[String]] = ListBuffer()
      cluster_words.foreach(words => {
        val wordCount = words._2.toList.sortBy(_._2).reverse.take(10)
        val result: ListBuffer[String] = ListBuffer()
        wordCount.foreach(word => result += word._1)
        topTenWords += result
      })


      var id = 0
      errors_map.foreach(e => {
        clusterInfo += Cluster(id + 1, size_map(id)._2, e._2, topTenWords(id).toList)
        id += 1
      })

      val results = Type("K-Means", WSSSE, clusterInfo.toList)
      implicit val format = Serialization.formats(NoTypeHints)
      val writer = new FileWriter(output_path)
      writer.write(Serialization.write(results))
      writer.close()


    }


    else if (algorithm.equals("B")) {

      val output_path = "Shivani_Shukla_Cluster_small_B_8_20.json"

      // Clustering the data into 6 clusters by BisectingKMeans.
      val bkm = new BisectingKMeans().setK(8).setSeed(42).setMaxIterations(20)
      val model = bkm.run(parsedData)

      val WSSSE = model.computeCost(parsedData)
      val centroids = model.clusterCenters

      val errors_map = documents.mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[(Int, Double)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = model.predict(tfidf)

          val newerr = euclideanDistance(centroids(cluster_id).toArray, tfidf.toArray)
          result.append((cluster_id, newerr))

          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).sortByKey().collect()

      val size_map = documents.mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[(Int, Int)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = model.predict(tfidf)


          result.append((cluster_id, 1))

          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).sortByKey().collect()


      val cluster_words = documents.repartition((1)).mapPartitions { document_list: Iterator[Seq[String]] =>
        val result: ListBuffer[((String, Int), Int)] = ListBuffer()

        for (document <- document_list) {
          val tf = hashingTF.transform(document)
          val tfidf = idf.transform(tf)
          val cluster_id = model.predict(tfidf)


          for (word <- document) {
            result.append(((word, cluster_id), 1))
          }
          result
        }

        result.toIterator
      }.reduceByKey((a, b) => a + b).map(row => (row._1._2, (row._1._1, row._2))).groupByKey().sortByKey().collect()


      val topTenWords: ListBuffer[ListBuffer[String]] = ListBuffer()
      cluster_words.foreach(words => {
        val wordCount = words._2.toList.sortBy(_._2).reverse.take(10)
        val result: ListBuffer[String] = ListBuffer()
        wordCount.foreach(word => result += word._1)
        topTenWords += result
      })


      var id = 0


      errors_map.foreach(e => {
        clusterInfo += Cluster(id + 1, size_map(id)._2, e._2, topTenWords(id).toList)
        id += 1
      })


      val results = Type("K-Means", WSSSE, clusterInfo.toList)
      implicit val format = Serialization.formats(NoTypeHints)
      val writer = new FileWriter(output_path)
      writer.write(Serialization.write(results))
      writer.close()

    }
  }
}

