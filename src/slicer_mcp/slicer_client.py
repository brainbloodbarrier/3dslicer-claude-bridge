"""HTTP client for 3D Slicer WebServer API."""

import json
import logging
import sys
from typing import Optional, Dict, List, Any

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging to stderr (stdout reserved for MCP)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr
)
logger = logging.getLogger("slicer-mcp")


class SlicerConnectionError(Exception):
    """Raised when connection to Slicer WebServer fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class SlicerClient:
    """HTTP client for 3D Slicer WebServer API.

    This client provides methods to interact with Slicer's REST API endpoints
    including Python execution, screenshot capture, scene inspection, and data loading.
    """

    def __init__(self, base_url: str = "http://localhost:2016", timeout: int = 30):
        """Initialize Slicer HTTP client.

        Args:
            base_url: Base URL of Slicer WebServer (default: http://localhost:2016)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Note: We use direct requests.get/post instead of Session to avoid
        # connection reuse issues with Slicer's WebServer (it closes connections
        # immediately after response, causing "Connection reset by peer" errors)

        logger.info(f"Initialized SlicerClient with base_url={self.base_url}, timeout={self.timeout}s")

    def health_check(self) -> Dict[str, Any]:
        """Check connection to Slicer WebServer.

        Returns:
            Dict with connection status and response time

        Raises:
            SlicerConnectionError: If Slicer is not reachable
        """
        try:
            import time
            start_time = time.time()

            response = requests.get(
                f"{self.base_url}/slicer/mrml",
                timeout=self.timeout
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            response.raise_for_status()

            logger.info(f"Health check successful: {elapsed_ms}ms")

            return {
                "connected": True,
                "webserver_url": self.base_url,
                "response_time_ms": elapsed_ms,
            }

        except (ConnectionError, Timeout) as e:
            logger.error(f"Health check failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "timeout": self.timeout,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Health check request failed: {e}")
            raise SlicerConnectionError(
                f"Slicer WebServer request failed: {str(e)}",
                details={"url": self.base_url}
            )

    def exec_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in Slicer's Python environment.

        Args:
            code: Python code to execute

        Returns:
            Dict with success status and result/output

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug(f"Executing Python code: {code[:100]}...")

            response = requests.post(
                f"{self.base_url}/slicer/exec",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.text

            logger.info(f"Python execution successful, result length: {len(result)}")

            return {
                "success": True,
                "result": result,
                "stdout": "",
                "stderr": ""
            }

        except (ConnectionError, Timeout) as e:
            logger.error(f"Python execution connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Python execution failed: {e}")
            raise SlicerConnectionError(
                f"Python execution failed: {str(e)}",
                details={"code": code[:200]}
            )

    def get_screenshot(self, view: str = "Red", scroll_to: Optional[float] = None) -> bytes:
        """Capture screenshot from a slice view.

        Args:
            view: View name - Red (axial), Yellow (sagittal), Green (coronal)
            scroll_to: Slice position from 0.0 to 1.0 (optional)

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            url = f"{self.base_url}/slicer/slice?view={view}"
            if scroll_to is not None:
                url += f"&scrollTo={scroll_to}"

            logger.debug(f"Capturing screenshot: view={view}, scroll_to={scroll_to}")

            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            image_bytes = response.content

            logger.info(f"Screenshot captured: {len(image_bytes)} bytes")

            return image_bytes

        except (ConnectionError, Timeout) as e:
            logger.error(f"Screenshot connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise SlicerConnectionError(
                f"Screenshot capture failed: {str(e)}",
                details={"view": view, "scroll_to": scroll_to}
            )

    def get_3d_screenshot(self, look_from_axis: Optional[str] = None) -> bytes:
        """Capture screenshot from 3D view.

        Args:
            look_from_axis: Camera axis - left, right, anterior, posterior, superior, inferior (optional)

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            url = f"{self.base_url}/slicer/threeD"
            if look_from_axis:
                url += f"?lookFromAxis={look_from_axis}"

            logger.debug(f"Capturing 3D screenshot: look_from_axis={look_from_axis}")

            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            image_bytes = response.content

            logger.info(f"3D screenshot captured: {len(image_bytes)} bytes")

            return image_bytes

        except (ConnectionError, Timeout) as e:
            logger.error(f"3D screenshot connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"3D screenshot capture failed: {e}")
            raise SlicerConnectionError(
                f"3D screenshot capture failed: {str(e)}",
                details={"look_from_axis": look_from_axis}
            )

    def get_full_screenshot(self) -> bytes:
        """Capture screenshot of full Slicer window.

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug("Capturing full window screenshot")

            response = requests.get(
                f"{self.base_url}/slicer/screenshot",
                timeout=self.timeout
            )
            response.raise_for_status()

            image_bytes = response.content

            logger.info(f"Full screenshot captured: {len(image_bytes)} bytes")

            return image_bytes

        except (ConnectionError, Timeout) as e:
            logger.error(f"Full screenshot connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Full screenshot capture failed: {e}")
            raise SlicerConnectionError(
                f"Full screenshot capture failed: {str(e)}"
            )

    def get_scene_nodes(self) -> List[Dict[str, Any]]:
        """Get list of all MRML scene nodes with IDs and names.

        Returns:
            List of node dictionaries with id, name, type

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug("Fetching scene nodes")

            # Get node names
            names_response = requests.get(
                f"{self.base_url}/slicer/mrml/names",
                timeout=self.timeout
            )
            names_response.raise_for_status()

            # Get node IDs
            ids_response = requests.get(
                f"{self.base_url}/slicer/mrml/ids",
                timeout=self.timeout
            )
            ids_response.raise_for_status()

            # Parse responses - Slicer returns JSON arrays
            names = json.loads(names_response.text)
            ids = json.loads(ids_response.text)

            # Combine into node list
            nodes = []
            for node_id, node_name in zip(ids, names):
                # Extract node type from ID (e.g., vtkMRMLScalarVolumeNode1 -> vtkMRMLScalarVolumeNode)
                node_type = ''.join(c for c in node_id if not c.isdigit())
                nodes.append({
                    "id": node_id,
                    "name": node_name,
                    "type": node_type
                })

            logger.info(f"Fetched {len(nodes)} scene nodes")

            return nodes

        except (ConnectionError, Timeout) as e:
            logger.error(f"Scene nodes connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Scene nodes fetch failed: {e}")
            raise SlicerConnectionError(
                f"Failed to fetch scene nodes: {str(e)}"
            )

    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """Get properties of a specific MRML node.

        Args:
            node_id: MRML node ID (e.g., vtkMRMLScalarVolumeNode1)

        Returns:
            Dict with node properties

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug(f"Fetching properties for node: {node_id}")

            response = requests.get(
                f"{self.base_url}/slicer/mrml/properties",
                params={"id": node_id},
                timeout=self.timeout
            )
            response.raise_for_status()

            # Properties are returned as text, parse as needed
            properties_text = response.text

            logger.info(f"Fetched properties for node {node_id}")

            # Return as dict (Slicer returns key=value pairs)
            properties = {}
            for line in properties_text.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    properties[key.strip()] = value.strip()

            return properties

        except (ConnectionError, Timeout) as e:
            logger.error(f"Node properties connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Node properties fetch failed: {e}")
            raise SlicerConnectionError(
                f"Failed to fetch node properties: {str(e)}",
                details={"node_id": node_id}
            )

    def load_sample_data(self, name: str) -> Dict[str, Any]:
        """Load a sample dataset into Slicer.

        Args:
            name: Sample dataset name (e.g., MRHead, CTChest)

        Returns:
            Dict with success status and message

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug(f"Loading sample data: {name}")

            response = requests.get(
                f"{self.base_url}/slicer/sampledata",
                params={"name": name},
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Sample data '{name}' loaded successfully")

            return {
                "success": True,
                "dataset_name": name,
                "message": f"Sample data '{name}' loaded successfully"
            }

        except (ConnectionError, Timeout) as e:
            logger.error(f"Sample data load connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Sample data load failed: {e}")
            raise SlicerConnectionError(
                f"Failed to load sample data '{name}': {str(e)}",
                details={
                    "dataset_name": name,
                    "suggestion": "Check dataset name is valid (MRHead, CTChest, CTACardio, DTIBrain, MRBrainTumor1, MRBrainTumor2)"
                }
            )

    def set_layout(self, layout: str, gui_mode: str = "full") -> Dict[str, Any]:
        """Set Slicer viewer layout and GUI mode.

        Args:
            layout: Layout name (FourUp, OneUp3D, OneUpRedSlice, Conventional, SideBySide)
            gui_mode: GUI mode - full (complete GUI) or viewers (viewers only)

        Returns:
            Dict with success status and message

        Raises:
            SlicerConnectionError: If request fails
        """
        try:
            logger.debug(f"Setting layout: {layout}, gui_mode: {gui_mode}")

            response = requests.get(
                f"{self.base_url}/slicer/gui",
                params={
                    "contents": gui_mode,
                    "viewersLayout": layout
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Layout changed to {layout} with {gui_mode} GUI mode")

            return {
                "success": True,
                "layout": layout,
                "gui_mode": gui_mode,
                "message": f"Layout changed to {layout}"
            }

        except (ConnectionError, Timeout) as e:
            logger.error(f"Layout change connection failed: {e}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    "url": self.base_url,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled"
                }
            )
        except RequestException as e:
            logger.error(f"Layout change failed: {e}")
            raise SlicerConnectionError(
                f"Failed to set layout: {str(e)}",
                details={
                    "layout": layout,
                    "gui_mode": gui_mode,
                    "suggestion": "Check layout name is valid (FourUp, OneUp3D, OneUpRedSlice, Conventional, SideBySide)"
                }
            )
