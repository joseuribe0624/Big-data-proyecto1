load csv with headers from 'file:///PTBR_edges.csv' as row
match (p:User)
where p.id = row.from
match (b:User)
where b.id = row.to
merge (p)-[r:FRIEND_WITH]-(b)
return count(row)
