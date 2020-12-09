MATCH (p:User)
WITH collect(p) AS people
CALL apoc.export.csv.data(people, [], "dataset_topo.csv", {})
YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data