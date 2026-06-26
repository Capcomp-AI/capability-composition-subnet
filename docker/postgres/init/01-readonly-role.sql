-- The role the candidate's SQL tool connects as.
--
-- Statement inspection in the tool is a filter; this is the enforcement. Even a
-- statement that got past the inspector cannot write, because the role has no
-- grant that would let it.
--
-- Schema-level SELECT grants are issued per instance when its snapshot is built,
-- so a role that somehow persisted across instances still could not read a
-- schema it was never granted.

CREATE ROLE capsub_readonly WITH LOGIN PASSWORD 'readonly-set-at-deploy';

REVOKE ALL ON SCHEMA public FROM capsub_readonly;
REVOKE CREATE ON DATABASE maintenance FROM PUBLIC;

ALTER ROLE capsub_readonly SET default_transaction_read_only = on;
ALTER ROLE capsub_readonly SET statement_timeout = '10s';
-- An idle transaction holding locks would stall the next instance's teardown.
ALTER ROLE capsub_readonly SET idle_in_transaction_session_timeout = '30s';
