// latest connector release
// spark-shell --conf spark.neo4j.bolt.url=bolt://localhost:7687 --packages graphframes:graphframes:0.8.1-spark3.0-s_2.12,neo4j-contrib:neo4j-connector-apache-spark_2.12:4.0.0

// Considerations: You should have PTBR_target.csv and PTBR_edges.csv into Neo4j's imports files.
// Exported CSV is going to be into Neo4j's imports files.

import org.neo4j.spark._

val neo = Neo4j(sc)

loaduser = neo.cypher("load csv with headers from 'file:///PTBR_target.csv' as row merge (p:User{id: row.new_id, days: row.days, mature: row.mature, views: row.views, partner: row.partner}) return count(row)").loadRowRdd
loaduser.show()

loadedges = neo.cypher("load csv with headers from 'file:///PTBR_edges.csv' as row match (p:User) where p.id = row.from match (b:User) where b.id = row.to merge (p)-[r:FRIEND_WITH]-(b) return count(row)").loadRowRdd
loadedges.show()

creategraph = neo.cypher("CALL gds.graph.create('twitch', ['User'], 'FRIEND_WITH');").loadRowRdd
creategraph.show()

betweenness = neo.cypher("CALL gds.betweenness.write('twitch', { writeProperty: 'betweenness' }) yield minimumScore, maximumScore, scoreSum, nodePropertiesWritten").loadRowRdd
betweenness.show()

pageRank = neo.cypher("CALL gds.pageRank.write('twitch', {writeProperty: 'pageRank',maxIterations: 20,dampingFactor: 0.85}) YIELD nodePropertiesWritten, ranIterations").loadRowRdd
pageRank.show()

louvain = neo.cypher("CALL gds.louvain.write('twitch', { writeProperty: 'louvain' }) YIELD communityCount, modularity, modularities").loadRowRdd
louvain.show()

triangles = neo.cypher("CALL gds.graph.create('twitch2', 'User', {FRIEND_WITH: {orientation: 'UNDIRECTED'}}) YIELD graphName, nodeCount, relationshipCount, createMillis; CALL gds.triangleCount.write('twitch2', { writeProperty: 'triangles'}) YIELD globalTriangleCount, nodeCount").loadRowRdd
triangles.show()
  

node2vec = neo.cypher("CALL gds.alpha.node2vec.write('twitch', {embeddingDimension: 2, writeProperty: 'node2vec'}) yield nodeCount, nodePropertiesWritten").loadRowRdd
node2vec.show()

save = neo.cypher("MATCH (p:User) WITH collect(p) AS people CALL apoc.export.csv.data(people, [], 'dataset_topo.csv', {}) YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data").loadRowRdd
save.show()