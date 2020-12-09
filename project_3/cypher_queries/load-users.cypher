load csv with headers from 'file:///PTBR_target.csv' as row
merge (p:User{id: row.new_id, days: row.days, mature: row.mature, views: row.views, partner: row.partner})
return count(row)
