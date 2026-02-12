import logging
import re

def create_rbac_tables(conn):
    """Create the retrieval_policies table and add RBAC columns to sheets_metadata."""
    commands = [
        """
        CREATE TABLE IF NOT EXISTS retrieval_policies (
            policy_id SERIAL PRIMARY KEY,
            role TEXT NOT NULL,
            data_domain TEXT NOT NULL,
            sensitivity_level TEXT NOT NULL,
            allowed BOOLEAN DEFAULT false,
            allow_aggregation BOOLEAN DEFAULT true,
            allow_raw_rows BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(role, data_domain, sensitivity_level)
        );
        """,
        # Seed default policies if the table is empty
        """
        INSERT INTO retrieval_policies (role, data_domain, sensitivity_level, allowed, allow_aggregation, allow_raw_rows)
        SELECT role, data_domain, sensitivity_level, allowed, allow_aggregation, allow_raw_rows
        FROM (VALUES
            ('CEO', 'finance', 'confidential', true, true, true),
            ('CEO', 'hr', 'confidential', true, true, true),
            ('Finance_Manager', 'finance', 'confidential', true, true, true),
            ('Finance_Manager', 'hr', 'confidential', false, false, false),
            ('Analyst', 'finance', 'confidential', true, true, false),
            ('Employee', 'finance', 'internal', true, true, false)
        ) AS t(role, data_domain, sensitivity_level, allowed, allow_aggregation, allow_raw_rows)
        WHERE NOT EXISTS (SELECT 1 FROM retrieval_policies LIMIT 1);
        """
    ]
    
    try:
        with conn.cursor() as cur:
            for command in commands:
                cur.execute(command)
                
            # Add sensitivity + domain columns to sheets_metadata if they don't exist
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='sheets_metadata'")
            sheet_cols = {row[0] for row in cur.fetchall()}
            
            if 'data_domain' not in sheet_cols:
                logging.info("Adding data_domain to sheets_metadata")
                cur.execute("ALTER TABLE sheets_metadata ADD COLUMN data_domain TEXT")
            
            if 'sensitivity_level' not in sheet_cols:
                logging.info("Adding sensitivity_level to sheets_metadata")
                cur.execute("ALTER TABLE sheets_metadata ADD COLUMN sensitivity_level TEXT")
            
            # Create index for faster filtering
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sheets_domain_sensitivity ON sheets_metadata(data_domain, sensitivity_level)")
                
        conn.commit()
        logging.info("RBAC tables and columns verified.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating RBAC tables: {e}")

def validate_sql_access(sql_query, sheet_info):
    """
    Runtime validation to enforce access rules even if LLM fails.
    Returns True if allowed, raises PermissionError otherwise.
    """
    allow_raw = sheet_info.get('allow_raw_rows', False)
    
    if not allow_raw:
        sql_lower = sql_query.lower()
        
        # Check for forbidden patterns (row-level access)
        # select * is dangerous, but even selective selection without aggregation is row-level
        forbidden_patterns = [
            r'select\s+\*',
            r'limit', # often used to peek at rows
            r'offset',
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, sql_lower):
                raise PermissionError(
                    f"🔒 Access Denied: Raw row-level queries not allowed for {sheet_info.get('data_domain', 'this')} data. "
                    f"Only aggregated summaries are permitted."
                )
        
        # Must contain at least one aggregation function or a GROUP BY
        required_patterns = [r'sum\(', r'avg\(', r'count\(', r'max\(', r'min\(', r'group\s+by']
        
        has_aggregation = any(re.search(pattern, sql_lower) for pattern in required_patterns)
        
        if not has_aggregation:
            raise PermissionError(
                f"🔒 Access Denied: You must use aggregation functions (SUM, COUNT, AVG) for this data."
            )
    
    return True
