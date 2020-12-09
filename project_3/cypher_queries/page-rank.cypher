CALL gds.pageRank.write("twitch", {
  writeProperty: 'pageRank',
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodePropertiesWritten, ranIterations
