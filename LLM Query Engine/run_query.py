import sys
from nlq_to_snowflake import nlq_to_snowflake

# Query to run
query = "Show me the top 5 stores with highest sales"

# Run the query
result = nlq_to_snowflake(query)
