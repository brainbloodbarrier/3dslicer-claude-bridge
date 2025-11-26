# GitHub Issues to Create

**Note**: Network connectivity issues prevented automatic issue creation. Create these manually via GitHub UI or `gh issue create` when connectivity is restored.

---

## Issue #1: Implement MCP server core with stdio transport

**Labels**: enhancement, priority-high

**Body**:
```markdown
## Description
Create the foundational MCP server using FastMCP framework with stdio transport.

## Tasks
- [ ] Initialize FastMCP server instance
- [ ] Implement MCP protocol handshake (initialize method)
- [ ] Implement tools/list endpoint
- [ ] Implement resources/list endpoint
- [ ] Configure stdio transport with proper JSON-RPC 2.0 messaging
- [ ] Add structured logging to stderr (avoiding stdout pollution)
- [ ] Create main entry point for uv run slicer-mcp

## Acceptance Criteria
- Server responds to initialize request with capabilities
- Server lists 4 tools and 3 resources
- Server communicates via stdio without errors
- Logs appear in stderr, not stdout

## Dependencies
None

## References
- SPECIFICATION.md
- ARCHITECTURE.md - Section 2 (FastMCP Framework)
```

**CLI Command**:
```bash
gh issue create --title "Implement MCP server core with stdio transport" --label "enhancement,priority-high" --body "$(cat <<'EOF'
## Description
Create the foundational MCP server using FastMCP framework with stdio transport.

## Tasks
- [ ] Initialize FastMCP server instance
- [ ] Implement MCP protocol handshake (initialize method)
- [ ] Implement tools/list endpoint
- [ ] Implement resources/list endpoint
- [ ] Configure stdio transport with proper JSON-RPC 2.0 messaging
- [ ] Add structured logging to stderr (avoiding stdout pollution)
- [ ] Create main entry point for uv run slicer-mcp

## Acceptance Criteria
- Server responds to initialize request with capabilities
- Server lists 4 tools and 3 resources
- Server communicates via stdio without errors
- Logs appear in stderr, not stdout

## Dependencies
None

## References
- SPECIFICATION.md
- ARCHITECTURE.md - Section 2 (FastMCP Framework)
EOF
)"
```

---

## Issue #2: Implement HTTP client for 3D Slicer WebServer API

**Labels**: enhancement

**Body**:
```markdown
## Description
Create SlicerClient class with connection pooling and comprehensive WebServer API integration.

## Tasks
- [ ] Create SlicerClient class with configurable base URL
- [ ] Implement session management (requests.Session for connection pooling)
- [ ] Implement exec_python() method for Python code execution
- [ ] Implement get_screenshot() method with base64 encoding
- [ ] Implement get_scene_nodes() method for MRML scene inspection
- [ ] Implement get_volume_info() method for volume metadata
- [ ] Add timeout handling (30 second default)
- [ ] Add error mapping (HTTP errors to domain error codes)
- [ ] Add health_check() method for connection verification

## Acceptance Criteria
- Client maintains persistent HTTP session
- All methods return properly typed responses
- Connection errors mapped to SLICER_CONNECTION_ERROR
- Timeout errors handled gracefully
- Client works with Slicer WebServer on localhost:2016

## Dependencies
None

## References
- SPECIFICATION.md - Prerequisites
- ARCHITECTURE.md - Section 3 (HTTP Client Design)
```

**CLI Command**:
```bash
gh issue create --title "Implement HTTP client for 3D Slicer WebServer API" --label "enhancement" --body "$(cat <<'EOF'
## Description
Create SlicerClient class with connection pooling and comprehensive WebServer API integration.

## Tasks
- [ ] Create SlicerClient class with configurable base URL
- [ ] Implement session management (requests.Session for connection pooling)
- [ ] Implement exec_python() method for Python code execution
- [ ] Implement get_screenshot() method with base64 encoding
- [ ] Implement get_scene_nodes() method for MRML scene inspection
- [ ] Implement get_volume_info() method for volume metadata
- [ ] Add timeout handling (30 second default)
- [ ] Add error mapping (HTTP errors to domain error codes)
- [ ] Add health_check() method for connection verification

## Acceptance Criteria
- Client maintains persistent HTTP session
- All methods return properly typed responses
- Connection errors mapped to SLICER_CONNECTION_ERROR
- Timeout errors handled gracefully
- Client works with Slicer WebServer on localhost:2016

## Dependencies
None

## References
- SPECIFICATION.md - Prerequisites
- ARCHITECTURE.md - Section 3 (HTTP Client Design)
EOF
)"
```

---

## Issue #3: Implement core MCP tools

**Labels**: enhancement

**Body**:
```markdown
## Description
Implement all 4 MCP tools: capture_screenshot, list_scene_nodes, execute_python, measure_volume.

## Tasks
- [ ] Implement capture_screenshot tool
  - [ ] Parameter validation (view_type, slice_offset)
  - [ ] Screenshot capture via SlicerClient
  - [ ] Base64 PNG encoding
  - [ ] Return metadata (dimensions, view_type)
- [ ] Implement list_scene_nodes tool
  - [ ] Query MRML scene via SlicerClient
  - [ ] Parse node types and properties
  - [ ] Return structured node list
- [ ] Implement execute_python tool
  - [ ] Code validation (syntax check)
  - [ ] Execution via SlicerClient
  - [ ] Capture stdout/stderr
  - [ ] Handle execution errors
- [ ] Implement measure_volume tool
  - [ ] Node ID validation
  - [ ] Volume calculation via Slicer Python API
  - [ ] Per-segment volume breakdown
  - [ ] Return volumes in mm³ and mL
- [ ] Add comprehensive docstrings for Claude Code
- [ ] Add error handling for all tools

## Acceptance Criteria
- All 4 tools callable from Claude Code
- Parameter validation works correctly
- Error messages are actionable
- Tools appear in tools/list response
- Documentation strings guide Claude's usage

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)

## References
- SPECIFICATION.md - Tools Specification
- ARCHITECTURE.md - MCP Tool Handlers
```

**CLI Command**:
```bash
gh issue create --title "Implement core MCP tools" --label "enhancement" --body "$(cat <<'EOF'
## Description
Implement all 4 MCP tools: capture_screenshot, list_scene_nodes, execute_python, measure_volume.

## Tasks
- [ ] Implement capture_screenshot tool
  - [ ] Parameter validation (view_type, slice_offset)
  - [ ] Screenshot capture via SlicerClient
  - [ ] Base64 PNG encoding
  - [ ] Return metadata (dimensions, view_type)
- [ ] Implement list_scene_nodes tool
  - [ ] Query MRML scene via SlicerClient
  - [ ] Parse node types and properties
  - [ ] Return structured node list
- [ ] Implement execute_python tool
  - [ ] Code validation (syntax check)
  - [ ] Execution via SlicerClient
  - [ ] Capture stdout/stderr
  - [ ] Handle execution errors
- [ ] Implement measure_volume tool
  - [ ] Node ID validation
  - [ ] Volume calculation via Slicer Python API
  - [ ] Per-segment volume breakdown
  - [ ] Return volumes in mm³ and mL
- [ ] Add comprehensive docstrings for Claude Code
- [ ] Add error handling for all tools

## Acceptance Criteria
- All 4 tools callable from Claude Code
- Parameter validation works correctly
- Error messages are actionable
- Tools appear in tools/list response
- Documentation strings guide Claude's usage

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)

## References
- SPECIFICATION.md - Tools Specification
- ARCHITECTURE.md - MCP Tool Handlers
EOF
)"
```

---

## Issue #4: Implement MCP resources for scene state

**Labels**: enhancement

**Body**:
```markdown
## Description
Implement all 3 MCP resources: slicer://scene, slicer://volumes, slicer://status.

## Tasks
- [ ] Implement slicer://scene resource
  - [ ] Query full MRML scene structure
  - [ ] Include node relationships/connections
  - [ ] Return scene metadata (modified time, node count)
- [ ] Implement slicer://volumes resource
  - [ ] Query all volume nodes (scalar, vector, tensor)
  - [ ] Include volume properties (dimensions, spacing, origin)
  - [ ] Include file paths if available
- [ ] Implement slicer://status resource
  - [ ] Check Slicer connection status
  - [ ] Return Slicer version
  - [ ] Include response time metrics
  - [ ] Report scene loaded state
- [ ] Add resource documentation for Claude Code
- [ ] Handle connection errors gracefully

## Acceptance Criteria
- All 3 resources accessible from Claude Code
- Resources return current/fresh data (no stale cache)
- Resources appear in resources/list response
- Status resource helps debug connection issues
- Scene/volumes resources enable informed decision-making

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)

## References
- SPECIFICATION.md - Resources Specification
- ARCHITECTURE.md - MCP Resource Handlers
```

**CLI Command**:
```bash
gh issue create --title "Implement MCP resources for scene state" --label "enhancement" --body "$(cat <<'EOF'
## Description
Implement all 3 MCP resources: slicer://scene, slicer://volumes, slicer://status.

## Tasks
- [ ] Implement slicer://scene resource
  - [ ] Query full MRML scene structure
  - [ ] Include node relationships/connections
  - [ ] Return scene metadata (modified time, node count)
- [ ] Implement slicer://volumes resource
  - [ ] Query all volume nodes (scalar, vector, tensor)
  - [ ] Include volume properties (dimensions, spacing, origin)
  - [ ] Include file paths if available
- [ ] Implement slicer://status resource
  - [ ] Check Slicer connection status
  - [ ] Return Slicer version
  - [ ] Include response time metrics
  - [ ] Report scene loaded state
- [ ] Add resource documentation for Claude Code
- [ ] Handle connection errors gracefully

## Acceptance Criteria
- All 3 resources accessible from Claude Code
- Resources return current/fresh data (no stale cache)
- Resources appear in resources/list response
- Status resource helps debug connection issues
- Scene/volumes resources enable informed decision-making

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)

## References
- SPECIFICATION.md - Resources Specification
- ARCHITECTURE.md - MCP Resource Handlers
EOF
)"
```

---

## Issue #5: Configure MCP server in Claude Code and validate integration

**Labels**: enhancement, testing

**Body**:
```markdown
## Description
Configure the MCP server in Claude Code's mcp.json and perform end-to-end integration testing.

## Tasks
- [ ] Update ~/.claude/mcp.json with slicer-bridge configuration
- [ ] Set correct absolute path to slicer-bridge directory
- [ ] Configure environment variables (SLICER_URL)
- [ ] Restart Claude Code to load configuration
- [ ] Verify tool discovery (all 4 tools visible)
- [ ] Verify resource discovery (all 3 resources visible)
- [ ] Test end-to-end workflow:
  - [ ] Claude can check Slicer status
  - [ ] Claude can list scene nodes
  - [ ] Claude can capture screenshots
  - [ ] Claude can execute Python code
  - [ ] Claude can measure volumes
- [ ] Document any configuration issues encountered
- [ ] Create troubleshooting guide

## Acceptance Criteria
- MCP server starts successfully when Claude Code launches
- All tools and resources discoverable
- Sample workflow completes without errors
- Configuration documented in README.md
- Troubleshooting guide covers common issues

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)
- Issue #3 (MCP tools)
- Issue #4 (MCP resources)

## References
- README.md - Installation and Usage
- SPECIFICATION.md - Prerequisites
```

**CLI Command**:
```bash
gh issue create --title "Configure MCP server in Claude Code and validate integration" --label "enhancement,testing" --body "$(cat <<'EOF'
## Description
Configure the MCP server in Claude Code's mcp.json and perform end-to-end integration testing.

## Tasks
- [ ] Update ~/.claude/mcp.json with slicer-bridge configuration
- [ ] Set correct absolute path to slicer-bridge directory
- [ ] Configure environment variables (SLICER_URL)
- [ ] Restart Claude Code to load configuration
- [ ] Verify tool discovery (all 4 tools visible)
- [ ] Verify resource discovery (all 3 resources visible)
- [ ] Test end-to-end workflow:
  - [ ] Claude can check Slicer status
  - [ ] Claude can list scene nodes
  - [ ] Claude can capture screenshots
  - [ ] Claude can execute Python code
  - [ ] Claude can measure volumes
- [ ] Document any configuration issues encountered
- [ ] Create troubleshooting guide

## Acceptance Criteria
- MCP server starts successfully when Claude Code launches
- All tools and resources discoverable
- Sample workflow completes without errors
- Configuration documented in README.md
- Troubleshooting guide covers common issues

## Dependencies
- Issue #1 (MCP server core)
- Issue #2 (SlicerClient)
- Issue #3 (MCP tools)
- Issue #4 (MCP resources)

## References
- README.md - Installation and Usage
- SPECIFICATION.md - Prerequisites
EOF
)"
```

---

## Issue #6: Complete documentation and usage examples

**Labels**: documentation

**Body**:
```markdown
## Description
Complete comprehensive documentation and create realistic usage examples for cerebellar lesion analysis workflow.

## Tasks
- [ ] Complete README.md
  - [ ] Add installation instructions
  - [ ] Add configuration examples
  - [ ] Add usage examples (5+ scenarios)
  - [ ] Add troubleshooting section
- [ ] Create examples/cerebellar_lesion_workflow.md
  - [ ] Document realistic neurosurgical planning workflow
  - [ ] Include Claude Code prompts for each step
  - [ ] Include expected tool calls and responses
  - [ ] Include example screenshots
- [ ] Add code comments and docstrings
  - [ ] All public methods documented
  - [ ] Tool descriptions optimized for Claude understanding
  - [ ] Resource descriptions explain use cases
- [ ] Create CONTRIBUTING.md
  - [ ] Development setup instructions
  - [ ] Testing guidelines
  - [ ] Code style requirements
- [ ] Update SPECIFICATION.md with learnings from implementation
- [ ] Update ARCHITECTURE.md with any design changes

## Acceptance Criteria
- README provides clear path from install to first use
- Examples demonstrate real-world medical imaging workflows
- Documentation enables new users to get started in <15 minutes
- All public APIs documented with examples
- Contributing guidelines enable external contributions

## Dependencies
- All implementation issues (#1-5)

## References
- README.md (to be completed)
- SPECIFICATION.md
- ARCHITECTURE.md
```

**CLI Command**:
```bash
gh issue create --title "Complete documentation and usage examples" --label "documentation" --body "$(cat <<'EOF'
## Description
Complete comprehensive documentation and create realistic usage examples for cerebellar lesion analysis workflow.

## Tasks
- [ ] Complete README.md
  - [ ] Add installation instructions
  - [ ] Add configuration examples
  - [ ] Add usage examples (5+ scenarios)
  - [ ] Add troubleshooting section
- [ ] Create examples/cerebellar_lesion_workflow.md
  - [ ] Document realistic neurosurgical planning workflow
  - [ ] Include Claude Code prompts for each step
  - [ ] Include expected tool calls and responses
  - [ ] Include example screenshots
- [ ] Add code comments and docstrings
  - [ ] All public methods documented
  - [ ] Tool descriptions optimized for Claude understanding
  - [ ] Resource descriptions explain use cases
- [ ] Create CONTRIBUTING.md
  - [ ] Development setup instructions
  - [ ] Testing guidelines
  - [ ] Code style requirements
- [ ] Update SPECIFICATION.md with learnings from implementation
- [ ] Update ARCHITECTURE.md with any design changes

## Acceptance Criteria
- README provides clear path from install to first use
- Examples demonstrate real-world medical imaging workflows
- Documentation enables new users to get started in <15 minutes
- All public APIs documented with examples
- Contributing guidelines enable external contributions

## Dependencies
- All implementation issues (#1-5)

## References
- README.md (to be completed)
- SPECIFICATION.md
- ARCHITECTURE.md
EOF
)"
```

---

## Manual Creation Instructions

If using GitHub CLI is not working due to network issues, you can:

1. **Via GitHub Web UI**:
   - Go to repository Issues tab
   - Click "New Issue"
   - Copy title and body from above
   - Add labels manually
   - Submit

2. **Via GitHub CLI** (when connectivity restored):
   - Run the CLI commands shown above
   - Or use: `gh issue create --web` to open browser

3. **Batch Creation Script**:
   Save each issue to a file (`issue-1.md`, etc.) and run:
   ```bash
   for i in {1..6}; do
     gh issue create --title "$(head -n1 issue-$i.md)" \
                      --body "$(tail -n+2 issue-$i.md)" \
                      --label "enhancement"
   done
   ```
