CALL gds.graph.create(
  'twitch2',
  'User',
  {
    FRIEND_WITH: {
      orientation: 'UNDIRECTED'
    }
  }
)
YIELD graphName, nodeCount, relationshipCount, createMillis;
CALL gds.triangleCount.write('twitch2', {
  writeProperty: 'triangles'
})
YIELD globalTriangleCount, nodeCount