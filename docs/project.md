This repo has :
- a rag project where openscience dataset is used for evaluation purpose.
- a vector_db is chucked and stored in google drive.
- a during infureces it uses the vector_db get all relvent chucks.

During inference :
- user query is sent to vector_db using cosine similarity search
- cosine similaritly search formula =( A.B) / (||A|| ||B||)
- In  words  Cosine similarity is the cosine of the angle between the vectors; that is, it is the dot product of the vectors divided by the product of their lengths.
- It is sent to llm for beautify things alot.
- It is much reliable process.
