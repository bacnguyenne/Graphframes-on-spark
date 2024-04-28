// Databricks notebook source
// MAGIC %md
// MAGIC # GraphFrames on Spark
// MAGIC ## Read data
// MAGIC link data: https://www.kaggle.com/datasets/nivethaks17/2017-ford-gobike-ridedata/data

// COMMAND ----------

import org.apache.spark.sql.SparkSession

val spark: SparkSession = SparkSession.builder()
  .appName("GraphFrames")
  .getOrCreate()

val tripData = spark.sql("SELECT * FROM trip")
val stationData = spark.sql("SELECT start_station_name, member_birth_year FROM trip").dropDuplicates("start_station_name")

// COMMAND ----------

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.DataFrame
val station = stationData.withColumn("member_birth_year", col("member_birth_year").cast("int"))
display(station)


// COMMAND ----------

display(tripData)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Creating GraphFrames

// COMMAND ----------

val edges: DataFrame = tripData.withColumnRenamed("start_station_name", "src").withColumnRenamed("end_station_name", "dst")
val vertices: DataFrame = station.withColumnRenamed("start_station_name", "id")


// COMMAND ----------

import org.graphframes.GraphFrame
val stationGraph: GraphFrame = GraphFrame(vertices, edges)
stationGraph.cache()

// COMMAND ----------

println("Total Number of Stations: " + stationGraph.vertices.count())
println("Total Number of Trips in Graph: " + stationGraph.edges.count())
println("Total Number of Trips in Original Data: " + tripData.count())

// COMMAND ----------

import org.apache.spark.sql.functions.desc

stationGraph.edges.groupBy("src", "dst").count().orderBy(desc("count")).show(10)

// COMMAND ----------

stationGraph.edges
  .filter(($"src" === "Golden Gate Ave at Polk St" && $"dst" === "Eureka Valley Recreation Center") || ($"dst" === "Golden Gate Ave at Polk St" && $"src" === "Eureka Valley Recreation Center"))
  .groupBy("src", "dst")
  .count()
  .orderBy(desc("count"))
  .show(10)


// COMMAND ----------

// MAGIC %md
// MAGIC ## Explore In-Degree and Out-Degree Metrics

// COMMAND ----------

val inDeg = stationGraph.inDegrees
display(inDeg.orderBy(desc("inDegree")))

// COMMAND ----------

val outDeg = stationGraph.outDegrees
display(outDeg.orderBy(desc("outDegree")))

// COMMAND ----------

val degreeRatio = inDeg.join(outDeg, "id").selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio")
degreeRatio.orderBy(desc("degreeRatio")).show(10)
degreeRatio.orderBy("degreeRatio").show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ## motifs

// COMMAND ----------

val motifs = stationGraph.find("(a)-[ab]->(b); (b)-[ba]->(a)")
display(motifs)

// COMMAND ----------

val filtered = motifs.filter("a.member_birth_year > 1990")
display(filtered)

// COMMAND ----------

val filterid = motifs.filter("b.id = 'Central Ave at Fell St'")
display(filterid)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Subgraphs

// COMMAND ----------

val g = stationGraph.filterEdges("member_gender = 'Male'").filterVertices("member_birth_year > 1990").dropIsolatedVertices()
display(g.edges)

// COMMAND ----------

// Complex triplet filters
val paths = g.find("(a)-[ab]->(b)").filter("ab.member_gender = 'Male'").filter("a.member_birth_year < b.member_birth_year")
val ab2 = paths.select("ab.src", "ab.dst", "ab.member_gender")
val g2 = GraphFrame(g.vertices, ab2)

// COMMAND ----------

display(g2.vertices)

// COMMAND ----------

// MAGIC %md
// MAGIC ## PageRank

// COMMAND ----------

val ranks = stationGraph.pageRank.resetProbability(0.15).maxIter(10).run()
ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").show(10)

// COMMAND ----------

val ranks2 = stationGraph.pageRank.resetProbability(0.15).tol(0.01).run()
display(ranks2.vertices)

// COMMAND ----------

val ranks3 = stationGraph.pageRank.resetProbability(0.15).maxIter(10).sourceId("Snow Park").run()
display(ranks3.vertices)

// COMMAND ----------

// MAGIC %md
// MAGIC ## BFS

// COMMAND ----------

val BFS = stationGraph.bfs.fromExpr("id = 'Snow Park'").toExpr("member_birth_year > 1980").run()
display(BFS)

// COMMAND ----------

val filteredBFS = stationGraph.bfs.fromExpr("member_birth_year < 1990").toExpr("member_birth_year > 1980").edgeFilter("member_gender = 'Male'").maxPathLength(3).run()
display(filteredBFS)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Connected components
// MAGIC

// COMMAND ----------

spark.sparkContext.setCheckpointDir("/tmp/checkpoints")

val minGraph = GraphFrame(vertices, edges.sample(false, 0.001))
val cc = minGraph.connectedComponents.run()

// Hiển thị các thành phần liên thông không phải là 0
display(cc.filter("component != 0"))

// COMMAND ----------

val stronglyCoCom = stationGraph.stronglyConnectedComponents.maxIter(10).run()
display(stronglyCoCom.orderBy("component"))

// COMMAND ----------


