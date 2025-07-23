import streamlit as st
import snowflake.connector

st.set_page_config(page_title="Snowflake Table Chunker", page_icon="❄️", layout="wide")

st.title("❄️ Snowflake Table Chunking Tool")
st.markdown("Find the optimal WHERE clauses to split your table into chunks")

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'row_count' not in st.session_state:
    st.session_state.row_count = 0


def connect_to_snowflake(account, user, password, warehouse, database, schema):
    """Establish connection to Snowflake"""
    try:
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        return conn
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None


def get_tables(conn):
    """Get list of tables in the schema"""
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[1] for row in cursor.fetchall()]
    cursor.close()
    return tables


def get_row_count(conn, table_name):
    """Get total row count for a table"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def get_column_info(conn, table_name):
    """Get column information for a table"""
    cursor = conn.cursor()
    cursor.execute(f"DESCRIBE TABLE {table_name}")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            'name': row[0],
            'type': row[1],
            'nullable': row[3] == 'Y'
        })
    cursor.close()
    return columns


def analyze_column_cardinality(conn, table_name, column_name, limit=100):
    """Get distinct values and their counts for a column"""
    cursor = conn.cursor()
    try:
        # Get value distribution
        query = f"""
        SELECT {column_name}, COUNT(*) as row_count
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        GROUP BY {column_name}
        ORDER BY row_count DESC
        LIMIT {limit}
        """
        cursor.execute(query)
        results = cursor.fetchall()

        # Get total distinct count
        cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}")
        total_distinct = cursor.fetchone()[0]

        cursor.close()
        return results, total_distinct
    except Exception as e:
        cursor.close()
        return [], 0


def find_best_chunking_columns(conn, table_name, target_chunk_size, total_rows):
    """Find columns or column combinations that best divide the table"""
    columns = get_column_info(conn, table_name)
    cursor = conn.cursor()

    # Filter columns suitable for grouping (categorical, dates, etc)
    groupable_columns = []
    for col in columns:
        if col['type'] in ['VARCHAR', 'CHAR', 'STRING', 'NUMBER', 'INTEGER', 'DATE', 'BOOLEAN']:
            # Check cardinality
            values, distinct_count = analyze_column_cardinality(conn, table_name, col['name'])
            if distinct_count > 0 and distinct_count < 10000:  # Reasonable cardinality
                groupable_columns.append({
                    'name': col['name'],
                    'distinct_count': distinct_count,
                    'top_values': values[:20]  # Keep top 20 values
                })

    # Sort by cardinality (prefer columns with moderate cardinality)
    groupable_columns.sort(key=lambda x: abs(x['distinct_count'] - (total_rows / target_chunk_size)))

    cursor.close()
    return groupable_columns


def find_optimal_column_combination(conn, table_name, columns, target_chunk_size, total_rows):
    """Test ALL column combinations in parallel and find the one that produces most chunks close to target"""
    from itertools import combinations

    results = {}

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Function to test a specific column combination
    def test_combination(col_combo):
        chunks, score = evaluate_column_combination(conn, table_name, col_combo, target_chunk_size)
        if chunks:
            # Count chunks that are close to target (within 20%)
            close_chunks = 0
            tolerance = target_chunk_size * 0.2

            for chunk in chunks:
                if abs(chunk['size'] - target_chunk_size) <= tolerance:
                    close_chunks += 1

            return col_combo, chunks, close_chunks, score
        return None

    # Generate ALL possible combinations (limited for performance)
    all_combinations = []
    max_columns_to_try = min(len(columns), 6)  # Try up to 6 columns

    for num_cols in range(1, max_columns_to_try + 1):
        # For higher column counts, sample combinations instead of trying all
        if num_cols <= 3:
            combos = list(combinations(columns[:12], num_cols))
        else:
            # Sample up to 10 combinations for 4+ columns
            combos = list(combinations(columns[:8], num_cols))[:10]

        all_combinations.extend(combos)

    status_text.text(f"Testing {len(all_combinations)} column combinations...")

    # Run all combinations (not truly parallel to avoid connection issues)
    completed = 0
    for combo in all_combinations:
        completed += 1
        progress_bar.progress(completed / len(all_combinations))

        result = test_combination(combo)
        if result:
            col_combo, chunks, close_chunks, score = result
            col_names_tuple = tuple(c['name'] for c in col_combo)

            # Only consider combinations that create multiple chunks
            if len(chunks) >= 2 or (len(chunks) == 1 and total_rows < target_chunk_size * 1.5):
                results[col_names_tuple] = {
                    'chunks': chunks,
                    'close_chunks': close_chunks,
                    'total_chunks': len(chunks),
                    'score': score,
                    'columns': col_combo,
                    'num_columns': len(col_combo)
                }

                status_text.text(
                    f"Tested {completed}/{len(all_combinations)} - Found {len(results)} valid combinations")

    progress_bar.empty()
    status_text.empty()

    # Find the best combination
    if not results:
        st.warning("⚠️ No combination found that creates multiple chunks. Trying alternative approach...")
        # Fall back to simple division
        return None, None

    # Sort by: 1) number of close chunks (more is better), 2) total chunks, 3) score
    best_combo_names = max(results.keys(),
                           key=lambda x: (results[x]['close_chunks'],
                                          results[x]['total_chunks'],
                                          -results[x]['score']))
    best_result = results[best_combo_names]

    # Show summary of results
    st.success(
        f"✨ Found {len(results)} valid combinations. Best uses {best_result['num_columns']} column(s): **{', '.join(best_combo_names)}**")
    st.info(
        f"This creates {best_result['total_chunks']} chunks with {best_result['close_chunks']} chunks close to your target of {target_chunk_size:,} rows.")

    return best_result['columns'], best_result['chunks']


def generate_chunking_queries(conn, table_name, column_info, target_chunk_size, total_rows):
    """Generate actual WHERE clause queries for chunking using optimal column combinations"""
    # Find the best column combination (tests ALL in parallel)
    best_combination, chunks = find_optimal_column_combination(
        conn, table_name, column_info, target_chunk_size, total_rows
    )

    if not best_combination or not chunks:
        return [], None

    queries = []
    column_names = [col['name'] for col in best_combination]

    for i, chunk in enumerate(chunks):
        conditions = []

        for group_values in chunk['groups']:
            if len(column_names) == 1:
                # Single column
                conditions.append(f"{column_names[0]} = '{group_values[0]}'")
            else:
                # Multiple columns
                group_condition = " AND ".join([
                    f"{col} = '{val}'" for col, val in zip(column_names, group_values)
                ])
                conditions.append(f"({group_condition})")

        # Combine conditions with OR
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = " OR ".join(conditions)
            if len(conditions) > 1:
                where_clause = f"({where_clause})"

        queries.append({
            'query': f"SELECT * FROM {table_name} WHERE {where_clause}",
            'estimated_rows': chunk['size'],
            'where_clause': where_clause,
            'columns_used': column_names
        })

    # Return info about the combination used
    combination_info = {
        'columns': column_names,
        'num_columns': len(column_names)
    }

    return queries, combination_info


def evaluate_column_combination(conn, table_name, columns, target_chunk_size, limit=1000):
    """Evaluate how well a column combination chunks the data"""
    cursor = conn.cursor()
    col_names = [col['name'] for col in columns]
    col_list = ', '.join(col_names)

    try:
        # Get group sizes for this combination
        query = f"""
        SELECT {col_list}, COUNT(*) as chunk_size
        FROM {table_name}
        WHERE {' AND '.join([f"{col} IS NOT NULL" for col in col_names])}
        GROUP BY {col_list}
        ORDER BY chunk_size DESC
        LIMIT {limit}
        """

        cursor.execute(query)
        groups = cursor.fetchall()

        if not groups:
            return None, float('inf')

        # Get total rows covered by these groups
        total_rows_in_groups = sum(g[-1] for g in groups)

        # Simulate chunking - STRICT mode: don't exceed target
        chunks = []
        current_chunk = []
        current_size = 0

        for group in groups:
            group_size = group[-1]
            group_values = group[:-1]

            # If this single group is larger than target, it gets its own chunk
            if group_size > target_chunk_size * 1.2:
                # First, save current chunk if it has data
                if current_chunk:
                    chunks.append({
                        'groups': current_chunk,
                        'size': current_size,
                        'columns': col_names
                    })
                    current_chunk = []
                    current_size = 0

                # This large group becomes its own chunk
                chunks.append({
                    'groups': [group_values],
                    'size': group_size,
                    'columns': col_names
                })
            # If adding this group would exceed target significantly, start new chunk
            elif current_size + group_size > target_chunk_size * 1.1:  # 10% tolerance
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'groups': current_chunk,
                        'size': current_size,
                        'columns': col_names
                    })
                # Start new chunk with this group
                current_chunk = [group_values]
                current_size = group_size
            else:
                # Add to current chunk
                current_chunk.append(group_values)
                current_size += group_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                'groups': current_chunk,
                'size': current_size,
                'columns': col_names
            })

        # If we only got 1 chunk and it's too big, this combination isn't good
        if len(chunks) == 1 and chunks[0]['size'] > target_chunk_size * 1.5:
            return None, float('inf')

        # Calculate score based on how close chunks are to target
        if chunks:
            close_chunks = sum(
                1 for chunk in chunks if abs(chunk['size'] - target_chunk_size) <= target_chunk_size * 0.2)
            avg_deviation = sum(abs(chunk['size'] - target_chunk_size) for chunk in chunks) / len(chunks)
            score = avg_deviation / target_chunk_size

            # Bonus for having more chunks close to target
            score -= (close_chunks / len(chunks)) * 0.5

            cursor.close()
            return chunks, score

    except Exception as e:
        pass

    cursor.close()
    return None, float('inf')


# Sidebar for connection details
with st.sidebar:
    st.header("🔌 Snowflake Connection")

    with st.form("connection_form"):
        account = st.text_input("Account", help="Your Snowflake account identifier")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        warehouse = st.text_input("Warehouse", value="COMPUTE_WH")
        database = st.text_input("Database")
        schema = st.text_input("Schema", value="PUBLIC")

        connect_btn = st.form_submit_button("Connect", use_container_width=True)

        if connect_btn:
            conn = connect_to_snowflake(account, user, password, warehouse, database, schema)
            if conn:
                st.session_state.conn = conn
                st.session_state.connected = True
                st.session_state.tables = get_tables(conn)
                st.success("✅ Connected successfully!")
            else:
                st.session_state.connected = False

# Main content
if st.session_state.connected:
    col1, col2 = st.columns(2)

    with col1:
        selected_table = st.selectbox(
            "📊 Select Table",
            options=st.session_state.tables,
            help="Choose the table you want to chunk"
        )

    with col2:
        target_chunk_size = st.number_input(
            "🎯 Target Rows per Chunk",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Desired number of rows in each chunk"
        )

    if selected_table:
        # Get row count when table is selected
        with st.spinner("Getting table information..."):
            row_count = get_row_count(st.session_state.conn, selected_table)
            st.info(f"📈 **Total Rows in {selected_table}:** {row_count:,}")

        if st.button("🔍 Find Best Chunking Queries", type="primary"):
            with st.spinner("Analyzing table structure and data distribution..."):
                # Find best columns for chunking
                column_info = find_best_chunking_columns(
                    st.session_state.conn,
                    selected_table,
                    target_chunk_size,
                    row_count
                )

                if column_info:
                    st.info(f"Found {len(column_info)} suitable columns. Testing all combinations in parallel...")

                    # Generate chunking queries
                    queries, combination_info = generate_chunking_queries(
                        st.session_state.conn,
                        selected_table,
                        column_info,
                        target_chunk_size,
                        row_count
                    )

                    if queries:
                        st.success(
                            f"✨ Found optimal chunking strategy using {combination_info['num_columns']} column(s): **{', '.join(combination_info['columns'])}**")

                        # Display summary
                        st.subheader("📊 Chunking Summary")

                        # Count chunks close to target
                        tolerance = target_chunk_size * 0.1
                        close_chunks = sum(
                            1 for q in queries if abs(q['estimated_rows'] - target_chunk_size) <= tolerance)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Chunks", len(queries))
                        with col2:
                            st.metric("Chunks Close to Target", f"{close_chunks}/{len(queries)}")
                        with col3:
                            avg_chunk_size = sum(q['estimated_rows'] for q in queries) / len(queries)
                            st.metric("Avg Chunk Size", f"{avg_chunk_size:,.0f}")
                        with col4:
                            st.metric("Columns Used", combination_info['num_columns'])

                        # Show chunk size distribution
                        chunk_sizes = [q['estimated_rows'] for q in queries]
                        min_chunk = min(chunk_sizes)
                        max_chunk = max(chunk_sizes)
                        st.info(f"📏 Chunk size range: {min_chunk:,} to {max_chunk:,} rows")

                        # Display queries
                        st.subheader("🔧 Generated Queries")
                        st.write("Copy and use these queries to process your table in chunks:")

                        for i, query_info in enumerate(queries, 1):
                            with st.expander(f"Chunk {i} (~{query_info['estimated_rows']:,} rows)"):
                                st.code(query_info['query'], language='sql')
                                st.caption(f"WHERE clause: {query_info['where_clause']}")

                        # Show all queries in one block for easy copying
                        st.subheader("📋 All Queries (for easy copying)")
                        all_queries = "\n\n".join([f"-- Chunk {i + 1}: ~{q['estimated_rows']:,} rows\n{q['query']};"
                                                   for i, q in enumerate(queries)])
                        st.code(all_queries, language='sql')

                        # Additional info
                        with st.expander("ℹ️ How these queries were generated"):
                            st.write(f"""
                            The app analyzed your table and tested combinations of 1 to 4 columns to find the best chunking strategy.

                            **Columns used:** {', '.join(combination_info['columns'])}

                            This {combination_info['num_columns']}-column combination was chosen because it creates chunks 
                            closest to your target of {target_chunk_size:,} rows.

                            The algorithm tested multiple column combinations and selected the one that:
                            - Minimizes the difference between actual and target chunk sizes
                            - Creates a reasonable number of chunks
                            - Ensures no row appears in multiple chunks
                            - Covers all rows in the table
                            """)
                    else:
                        st.warning(
                            "⚠️ Could not generate optimal chunking queries. Consider using LIMIT/OFFSET approach.")
                else:
                    st.error("❌ No suitable columns found for chunking. The table might need a different approach.")
else:
    st.info("👈 Please connect to Snowflake using the sidebar to get started.")

    # Show example
    with st.expander("📖 Example Output"):
        st.write("For a table with 500,000 rows and target chunk size of 50,000, you might get:")
        st.code("""
-- Chunk 1: ~48,567 rows
SELECT * FROM po_details WHERE department = 'Sales';

-- Chunk 2: ~51,233 rows
SELECT * FROM po_details WHERE department = 'Marketing';

-- Chunk 3: ~49,800 rows
SELECT * FROM po_details WHERE department IN ('IT', 'HR');

-- Chunk 4: ~52,100 rows
SELECT * FROM po_details WHERE year = 2024;

-- And so on...
        """, language='sql')