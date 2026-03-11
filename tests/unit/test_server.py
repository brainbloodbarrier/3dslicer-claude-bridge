"""Unit tests for server.py — tool wrappers, error handling, resources, and main().

Covers all 46 @mcp.tool() wrappers (success + error paths),
_handle_tool_error() with all 5 exception branches,
4 @mcp.resource() wrappers, and main() startup paths.
"""

from unittest.mock import patch

import pytest

from slicer_mcp import server
from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.server import _handle_tool_error, main

# ============================================================================
# Shared constants
# ============================================================================

_OK = {"success": True, "result": "ok"}


# ============================================================================
# _handle_tool_error — all 5 exception branches
# ============================================================================


class TestHandleToolError:
    """Test _handle_tool_error maps every exception type to the correct dict."""

    def test_validation_error(self):
        err = ValidationError("bad value", field="node_id", value="!!!")
        result = _handle_tool_error(err, "test_tool")
        assert result == {
            "success": False,
            "error": "bad value",
            "error_type": "validation",
            "field": "node_id",
            "value": "!!!",
        }

    def test_circuit_open_error(self):
        err = CircuitOpenError("breaker open", "slicer", 30.0)
        result = _handle_tool_error(err, "test_tool")
        assert result["success"] is False
        assert result["error_type"] == "circuit_open"
        assert "breaker open" in result["error"]

    def test_timeout_error(self):
        err = SlicerTimeoutError("timed out", details={"timeout_s": 30})
        result = _handle_tool_error(err, "test_tool")
        assert result == {
            "success": False,
            "error": "timed out",
            "error_type": "timeout",
            "details": {"timeout_s": 30},
        }

    def test_connection_error(self):
        err = SlicerConnectionError("refused", details={"url": "http://x"})
        result = _handle_tool_error(err, "test_tool")
        assert result == {
            "success": False,
            "error": "refused",
            "error_type": "connection",
            "details": {"url": "http://x"},
        }

    def test_unexpected_error(self):
        err = RuntimeError("kaboom")
        result = _handle_tool_error(err, "test_tool")
        assert result == {
            "success": False,
            "error": "kaboom",
            "error_type": "unexpected",
        }

    def test_timeout_error_no_details(self):
        """SlicerTimeoutError with default empty details."""
        err = SlicerTimeoutError("timed out")
        result = _handle_tool_error(err, "t")
        assert result["details"] == {}

    def test_connection_error_no_details(self):
        """SlicerConnectionError with default empty details."""
        err = SlicerConnectionError("refused")
        result = _handle_tool_error(err, "t")
        assert result["details"] == {}


# ============================================================================
# Parametrized data for all 46 tool wrappers
# ============================================================================

# Each entry: (server_function, patch_target, kwargs_for_call)
_WRAPPERS = [
    # --- base_tools (server.tools.*) ---
    pytest.param(
        server.capture_screenshot,
        "slicer_mcp.server.tools.capture_screenshot",
        {"view_type": "axial"},
        id="capture_screenshot",
    ),
    pytest.param(
        server.list_scene_nodes,
        "slicer_mcp.server.tools.list_scene_nodes",
        {},
        id="list_scene_nodes",
    ),
    pytest.param(
        server.execute_python,
        "slicer_mcp.server.tools.execute_python",
        {"code": "x = 1"},
        id="execute_python",
    ),
    pytest.param(
        server.measure_volume,
        "slicer_mcp.server.tools.measure_volume",
        {"node_id": "vtkMRMLSegmentationNode1"},
        id="measure_volume",
    ),
    pytest.param(
        server.list_sample_data,
        "slicer_mcp.server.tools.list_sample_data",
        {},
        id="list_sample_data",
    ),
    pytest.param(
        server.load_sample_data,
        "slicer_mcp.server.tools.load_sample_data",
        {"dataset_name": "MRHead"},
        id="load_sample_data",
    ),
    pytest.param(
        server.set_layout,
        "slicer_mcp.server.tools.set_layout",
        {"layout": "FourUp"},
        id="set_layout",
    ),
    pytest.param(
        server.import_dicom,
        "slicer_mcp.server.tools.import_dicom",
        {"folder_path": "/tmp/dicom"},
        id="import_dicom",
    ),
    pytest.param(
        server.list_dicom_studies,
        "slicer_mcp.server.tools.list_dicom_studies",
        {},
        id="list_dicom_studies",
    ),
    pytest.param(
        server.list_dicom_series,
        "slicer_mcp.server.tools.list_dicom_series",
        {"study_uid": "1.2.3"},
        id="list_dicom_series",
    ),
    pytest.param(
        server.load_dicom_series,
        "slicer_mcp.server.tools.load_dicom_series",
        {"series_uid": "1.2.3.4"},
        id="load_dicom_series",
    ),
    pytest.param(
        server.run_brain_extraction,
        "slicer_mcp.server.tools.run_brain_extraction",
        {"input_node_id": "vtkMRMLScalarVolumeNode1"},
        id="run_brain_extraction",
    ),
    # --- X-ray diagnostics (server.diagnostic_tools_xray.*) ---
    pytest.param(
        server.measure_sagittal_balance_xray,
        "slicer_mcp.server.diagnostic_tools_xray.measure_sagittal_balance_xray",
        {"volume_node_id": "v1", "landmarks": {}},
        id="measure_sagittal_balance_xray",
    ),
    pytest.param(
        server.measure_coronal_balance_xray,
        "slicer_mcp.server.diagnostic_tools_xray.measure_coronal_balance_xray",
        {"volume_node_id": "v1", "landmarks": {}},
        id="measure_coronal_balance_xray",
    ),
    pytest.param(
        server.measure_listhesis_dynamic_xray,
        "slicer_mcp.server.diagnostic_tools_xray.measure_listhesis_dynamic_xray",
        {
            "volume_node_ids": {"neutral": "n1"},
            "landmarks_per_position": {},
            "levels": ["L4-L5"],
        },
        id="measure_listhesis_dynamic_xray",
    ),
    pytest.param(
        server.detect_vertebral_fractures_xray,
        "slicer_mcp.server.diagnostic_tools_xray.detect_vertebral_fractures_xray",
        {"volume_node_id": "v1", "landmarks_per_vertebra": {}},
        id="detect_vertebral_fractures_xray",
    ),
    pytest.param(
        server.measure_cobb_angle_xray,
        "slicer_mcp.server.diagnostic_tools_xray.measure_cobb_angle_xray",
        {"volume_node_id": "v1", "landmarks": {}},
        id="measure_cobb_angle_xray",
    ),
    pytest.param(
        server.classify_disc_degeneration_xray,
        "slicer_mcp.server.diagnostic_tools_xray.classify_disc_degeneration_xray",
        {"volume_node_id": "v1", "landmarks_per_disc": {}},
        id="classify_disc_degeneration_xray",
    ),
    # --- CT diagnostics (server.diagnostic_tools_ct.*) ---
    pytest.param(
        server.detect_vertebral_fractures_ct,
        "slicer_mcp.server.diagnostic_tools_ct.detect_vertebral_fractures_ct",
        {"volume_node_id": "v1"},
        id="detect_vertebral_fractures_ct",
    ),
    pytest.param(
        server.assess_osteoporosis_ct,
        "slicer_mcp.server.diagnostic_tools_ct.assess_osteoporosis_ct",
        {"volume_node_id": "v1"},
        id="assess_osteoporosis_ct",
    ),
    pytest.param(
        server.detect_metastatic_lesions_ct,
        "slicer_mcp.server.diagnostic_tools_ct.detect_metastatic_lesions_ct",
        {"volume_node_id": "v1"},
        id="detect_metastatic_lesions_ct",
    ),
    pytest.param(
        server.calculate_sins_score,
        "slicer_mcp.server.diagnostic_tools_ct.calculate_sins_score",
        {"volume_node_id": "v1"},
        id="calculate_sins_score",
    ),
    pytest.param(
        server.measure_listhesis_ct,
        "slicer_mcp.server.diagnostic_tools_ct.measure_listhesis_ct",
        {"volume_node_id": "v1"},
        id="measure_listhesis_ct",
    ),
    pytest.param(
        server.measure_spinal_canal_ct,
        "slicer_mcp.server.diagnostic_tools_ct.measure_spinal_canal_ct",
        {"volume_node_id": "v1"},
        id="measure_spinal_canal_ct",
    ),
    # --- Spine tools (server.spine_tools.*) ---
    pytest.param(
        server.measure_ccj_angles,
        "slicer_mcp.server.spine_tools.measure_ccj_angles",
        {"segmentation_node_id": "seg1"},
        id="measure_ccj_angles",
    ),
    pytest.param(
        server.measure_spine_alignment,
        "slicer_mcp.server.spine_tools.measure_spine_alignment",
        {"segmentation_node_id": "seg1"},
        id="measure_spine_alignment",
    ),
    pytest.param(
        server.segment_spine,
        "slicer_mcp.server.spine_tools.segment_spine",
        {"input_node_id": "v1"},
        id="segment_spine",
    ),
    pytest.param(
        server.visualize_spine_segmentation,
        "slicer_mcp.server.spine_tools.visualize_spine_segmentation",
        {
            "segmentation_node_id": "seg1",
            "volume_node_id": "v1",
            "output_path": "/tmp/out.png",
        },
        id="visualize_spine_segmentation",
    ),
    pytest.param(
        server.segment_vertebral_artery,
        "slicer_mcp.server.spine_tools.segment_vertebral_artery",
        {"input_node_id": "v1"},
        id="segment_vertebral_artery",
    ),
    pytest.param(
        server.analyze_bone_quality,
        "slicer_mcp.server.spine_tools.analyze_bone_quality",
        {"input_node_id": "v1", "segmentation_node_id": "seg1"},
        id="analyze_bone_quality",
    ),
    # --- Instrumentation (server.instrumentation_tools.*) ---
    pytest.param(
        server.plan_cervical_screws,
        "slicer_mcp.server.instrumentation_tools.plan_cervical_screws",
        {
            "technique": "pedicle",
            "level": "C5",
            "segmentation_node_id": "seg1",
        },
        id="plan_cervical_screws",
    ),
    # --- MRI diagnostics (server.diagnostic_tools_mri.*) ---
    pytest.param(
        server.classify_modic_changes,
        "slicer_mcp.server.diagnostic_tools_mri.classify_modic_changes",
        {"t1_node_id": "t1", "t2_node_id": "t2"},
        id="classify_modic_changes",
    ),
    pytest.param(
        server.assess_disc_degeneration_mri,
        "slicer_mcp.server.diagnostic_tools_mri.assess_disc_degeneration_mri",
        {"t2_node_id": "t2"},
        id="assess_disc_degeneration_mri",
    ),
    pytest.param(
        server.detect_cord_compression_mri,
        "slicer_mcp.server.diagnostic_tools_mri.detect_cord_compression_mri",
        {"t2_node_id": "t2"},
        id="detect_cord_compression_mri",
    ),
    pytest.param(
        server.detect_metastatic_lesions_mri,
        "slicer_mcp.server.diagnostic_tools_mri.detect_metastatic_lesions_mri",
        {"t1_node_id": "t1", "t2_stir_node_id": "stir"},
        id="detect_metastatic_lesions_mri",
    ),
    # --- Workflow (server.workflow_modic.*) ---
    pytest.param(
        server.workflow_modic_eval,
        "slicer_mcp.server.workflow_modic.workflow_modic_eval",
        {"t1_volume_id": "t1", "t2_volume_id": "t2"},
        id="workflow_modic_eval",
    ),
    # --- Registration (server.registration_tools.*) ---
    pytest.param(
        server.place_landmarks,
        "slicer_mcp.server.registration_tools.place_landmarks",
        {"name": "Landmarks", "points": [[0, 0, 0]]},
        id="place_landmarks",
    ),
    pytest.param(
        server.get_landmarks,
        "slicer_mcp.server.registration_tools.get_landmarks",
        {"node_id": "vtkMRMLMarkupsFiducialNode1"},
        id="get_landmarks",
    ),
    pytest.param(
        server.register_volumes,
        "slicer_mcp.server.registration_tools.register_volumes",
        {"fixed_node_id": "f1", "moving_node_id": "m1"},
        id="register_volumes",
    ),
    pytest.param(
        server.register_landmarks,
        "slicer_mcp.server.registration_tools.register_landmarks",
        {"fixed_landmarks_id": "f1", "moving_landmarks_id": "m1"},
        id="register_landmarks",
    ),
    pytest.param(
        server.apply_transform,
        "slicer_mcp.server.registration_tools.apply_transform",
        {"node_id": "n1", "transform_node_id": "t1"},
        id="apply_transform",
    ),
    # --- Rendering (server.rendering_tools.*) ---
    pytest.param(
        server.enable_volume_rendering,
        "slicer_mcp.server.rendering_tools.enable_volume_rendering",
        {"node_id": "v1"},
        id="enable_volume_rendering",
    ),
    pytest.param(
        server.set_volume_rendering_property,
        "slicer_mcp.server.rendering_tools.set_volume_rendering_property",
        {"node_id": "v1"},
        id="set_volume_rendering_property",
    ),
    pytest.param(
        server.export_model,
        "slicer_mcp.server.rendering_tools.export_model",
        {
            "node_id": "m1",
            "output_directory": "/tmp",
            "filename": "model",
        },
        id="export_model",
    ),
    pytest.param(
        server.segmentation_to_models,
        "slicer_mcp.server.rendering_tools.segmentation_to_models",
        {"segmentation_node_id": "seg1"},
        id="segmentation_to_models",
    ),
    pytest.param(
        server.capture_3d_view,
        "slicer_mcp.server.rendering_tools.capture_3d_view",
        {"output_path": "/tmp/view.png"},
        id="capture_3d_view",
    ),
]


# ============================================================================
# Tool wrapper success path — all 46 wrappers
# ============================================================================


class TestToolWrapperSuccess:
    """Verify each wrapper delegates to its feature function and returns the result."""

    @pytest.mark.parametrize("fn,target,kwargs", _WRAPPERS)
    def test_returns_feature_result(self, fn, target, kwargs):
        with patch(target) as mock:
            mock.return_value = _OK
            assert fn(**kwargs) == _OK
            mock.assert_called_once()


# ============================================================================
# Tool wrapper error path — all 46 wrappers
# ============================================================================


class TestToolWrapperError:
    """Verify each wrapper catches exceptions and returns error dicts."""

    @pytest.mark.parametrize("fn,target,kwargs", _WRAPPERS)
    def test_connection_error(self, fn, target, kwargs):
        with patch(target) as mock:
            mock.side_effect = SlicerConnectionError("fail")
            result = fn(**kwargs)
            assert result["success"] is False
            assert result["error_type"] == "connection"


# ============================================================================
# Extra error-type coverage per module group
# ============================================================================


class TestToolWrapperErrorTypes:
    """Test different exception types propagate correctly through each module."""

    def test_base_tool_validation_error(self):
        err = ValidationError("bad", field="view_type", value="nope")
        with patch("slicer_mcp.server.tools.capture_screenshot") as mock:
            mock.side_effect = err
            result = server.capture_screenshot(view_type="nope")
            assert result["error_type"] == "validation"
            assert result["field"] == "view_type"

    def test_base_tool_timeout_error(self):
        with patch("slicer_mcp.server.tools.execute_python") as mock:
            mock.side_effect = SlicerTimeoutError("slow")
            result = server.execute_python(code="x=1")
            assert result["error_type"] == "timeout"

    def test_base_tool_circuit_open_error(self):
        with patch("slicer_mcp.server.tools.list_scene_nodes") as mock:
            mock.side_effect = CircuitOpenError("open", "slicer", 30)
            result = server.list_scene_nodes()
            assert result["error_type"] == "circuit_open"

    def test_xray_tool_validation_error(self):
        err = ValidationError("bad", field="volume_node_id", value="")
        target = "slicer_mcp.server.diagnostic_tools_xray.measure_sagittal_balance_xray"
        with patch(target) as mock:
            mock.side_effect = err
            result = server.measure_sagittal_balance_xray(volume_node_id="", landmarks={})
            assert result["error_type"] == "validation"

    def test_ct_tool_timeout_error(self):
        target = "slicer_mcp.server.diagnostic_tools_ct.detect_vertebral_fractures_ct"
        with patch(target) as mock:
            mock.side_effect = SlicerTimeoutError("slow")
            result = server.detect_vertebral_fractures_ct(volume_node_id="v1")
            assert result["error_type"] == "timeout"

    def test_spine_tool_unexpected_error(self):
        with patch("slicer_mcp.server.spine_tools.segment_spine") as mock:
            mock.side_effect = RuntimeError("unexpected")
            result = server.segment_spine(input_node_id="v1")
            assert result["error_type"] == "unexpected"

    def test_mri_tool_circuit_open_error(self):
        target = "slicer_mcp.server.diagnostic_tools_mri.classify_modic_changes"
        with patch(target) as mock:
            mock.side_effect = CircuitOpenError("open", "slicer", 30)
            result = server.classify_modic_changes(t1_node_id="t1", t2_node_id="t2")
            assert result["error_type"] == "circuit_open"

    def test_registration_tool_validation_error(self):
        err = ValidationError("bad", field="name", value="")
        target = "slicer_mcp.server.registration_tools.place_landmarks"
        with patch(target) as mock:
            mock.side_effect = err
            result = server.place_landmarks(name="", points=[[0, 0, 0]])
            assert result["error_type"] == "validation"

    def test_rendering_tool_timeout_error(self):
        target = "slicer_mcp.server.rendering_tools.enable_volume_rendering"
        with patch(target) as mock:
            mock.side_effect = SlicerTimeoutError("slow")
            result = server.enable_volume_rendering(node_id="v1")
            assert result["error_type"] == "timeout"

    def test_workflow_tool_connection_error(self):
        target = "slicer_mcp.server.workflow_modic.workflow_modic_eval"
        with patch(target) as mock:
            mock.side_effect = SlicerConnectionError("refused")
            result = server.workflow_modic_eval(t1_volume_id="t1", t2_volume_id="t2")
            assert result["error_type"] == "connection"

    def test_instrumentation_tool_unexpected_error(self):
        target = "slicer_mcp.server.instrumentation_tools.plan_cervical_screws"
        with patch(target) as mock:
            mock.side_effect = TypeError("wrong type")
            result = server.plan_cervical_screws(
                technique="pedicle",
                level="C5",
                segmentation_node_id="seg1",
            )
            assert result["error_type"] == "unexpected"
            assert "wrong type" in result["error"]


# ============================================================================
# Resource wrappers
# ============================================================================


class TestResourceWrappers:
    """Test all 4 @mcp.resource() wrappers delegate correctly."""

    def test_get_scene(self):
        with patch("slicer_mcp.server.resources.get_scene_resource") as mock:
            mock.return_value = '{"scene": "data"}'
            result = server.get_scene()
            assert result == '{"scene": "data"}'
            mock.assert_called_once()

    def test_get_volumes(self):
        with patch("slicer_mcp.server.resources.get_volumes_resource") as mock:
            mock.return_value = '{"volumes": []}'
            result = server.get_volumes()
            assert result == '{"volumes": []}'
            mock.assert_called_once()

    def test_get_status(self):
        with patch("slicer_mcp.server.resources.get_status_resource") as mock:
            mock.return_value = '{"connected": true}'
            result = server.get_status()
            assert result == '{"connected": true}'
            mock.assert_called_once()

    def test_get_workflows(self):
        with patch("slicer_mcp.server.resources.get_workflows_resource") as mock:
            mock.return_value = '{"workflows": []}'
            result = server.get_workflows()
            assert result == '{"workflows": []}'
            mock.assert_called_once()


# ============================================================================
# main() entry point
# ============================================================================


class TestMain:
    """Test main() startup, shutdown, and error paths."""

    def test_main_runs_server(self):
        with patch("slicer_mcp.server.mcp") as mock_mcp:
            mock_mcp.run.return_value = None
            main()
            mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_main_handles_keyboard_interrupt(self):
        with patch("slicer_mcp.server.mcp") as mock_mcp:
            mock_mcp.run.side_effect = KeyboardInterrupt()
            # Should not raise
            main()

    def test_main_propagates_exceptions(self):
        with patch("slicer_mcp.server.mcp") as mock_mcp:
            mock_mcp.run.side_effect = RuntimeError("fatal")
            with pytest.raises(RuntimeError, match="fatal"):
                main()


# ============================================================================
# Tool wrapper with optional parameters — verify defaults pass through
# ============================================================================


class TestToolWrapperDefaults:
    """Verify wrappers pass default values to feature functions."""

    def test_capture_screenshot_defaults(self):
        with patch("slicer_mcp.server.tools.capture_screenshot") as mock:
            mock.return_value = _OK
            server.capture_screenshot(view_type="3d")
            mock.assert_called_once_with("3d", None, None)

    def test_capture_screenshot_all_args(self):
        with patch("slicer_mcp.server.tools.capture_screenshot") as mock:
            mock.return_value = _OK
            server.capture_screenshot(
                view_type="axial",
                scroll_position=0.5,
                look_from_axis="left",
            )
            mock.assert_called_once_with("axial", 0.5, "left")

    def test_measure_volume_with_segment(self):
        with patch("slicer_mcp.server.tools.measure_volume") as mock:
            mock.return_value = _OK
            server.measure_volume(node_id="seg1", segment_name="Tumor")
            mock.assert_called_once_with("seg1", "Tumor")

    def test_set_layout_with_gui_mode(self):
        with patch("slicer_mcp.server.tools.set_layout") as mock:
            mock.return_value = _OK
            server.set_layout(layout="OneUp3D", gui_mode="viewers")
            mock.assert_called_once_with("OneUp3D", "viewers")

    def test_run_brain_extraction_all_args(self):
        with patch("slicer_mcp.server.tools.run_brain_extraction") as mock:
            mock.return_value = _OK
            server.run_brain_extraction(input_node_id="v1", method="swiss", device="cpu")
            mock.assert_called_once_with("v1", "swiss", "cpu")

    def test_detect_vertebral_fractures_ct_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_ct.detect_vertebral_fractures_ct"
        with patch(target) as mock:
            mock.return_value = _OK
            server.detect_vertebral_fractures_ct(
                volume_node_id="v1",
                segmentation_node_id="seg1",
                region="lumbar",
                classification_system="all",
            )
            mock.assert_called_once_with("v1", "seg1", "lumbar", "all")

    def test_assess_osteoporosis_ct_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_ct.assess_osteoporosis_ct"
        with patch(target) as mock:
            mock.return_value = _OK
            server.assess_osteoporosis_ct(
                volume_node_id="v1",
                segmentation_node_id="seg1",
                levels=["L1", "L2"],
                method="both",
            )
            mock.assert_called_once_with("v1", "seg1", ["L1", "L2"], "both")

    def test_segment_spine_all_args(self):
        with patch("slicer_mcp.server.spine_tools.segment_spine") as mock:
            mock.return_value = _OK
            server.segment_spine(
                input_node_id="v1",
                region="cervical",
                include_discs=True,
                include_spinal_cord=True,
            )
            mock.assert_called_once_with("v1", "cervical", True, True)

    def test_plan_cervical_screws_all_args(self):
        target = "slicer_mcp.server.instrumentation_tools.plan_cervical_screws"
        with patch(target) as mock:
            mock.return_value = _OK
            server.plan_cervical_screws(
                technique="lateral_mass",
                level="C4",
                segmentation_node_id="seg1",
                side="left",
                va_node_id="va1",
                variant="magerl",
                screw_diameter_mm=3.5,
                screw_length_mm=14.0,
            )
            mock.assert_called_once_with(
                technique="lateral_mass",
                level="C4",
                segmentation_node_id="seg1",
                side="left",
                va_node_id="va1",
                variant="magerl",
                screw_diameter_mm=3.5,
                screw_length_mm=14.0,
            )

    def test_register_volumes_all_args(self):
        target = "slicer_mcp.server.registration_tools.register_volumes"
        with patch(target) as mock:
            mock.return_value = _OK
            server.register_volumes(
                fixed_node_id="f1",
                moving_node_id="m1",
                transform_type="Affine",
                init_mode="useGeometryAlign",
                sampling_percentage=0.05,
                histogram_match=True,
                create_resampled=True,
            )
            mock.assert_called_once_with(
                "f1",
                "m1",
                "Affine",
                "useGeometryAlign",
                0.05,
                True,
                True,
            )

    def test_export_model_with_format(self):
        target = "slicer_mcp.server.rendering_tools.export_model"
        with patch(target) as mock:
            mock.return_value = _OK
            server.export_model(
                node_id="m1",
                output_directory="/tmp",
                filename="mesh",
                file_format="OBJ",
            )
            mock.assert_called_once_with("m1", "/tmp", "mesh", "OBJ")

    def test_capture_3d_view_all_args(self):
        target = "slicer_mcp.server.rendering_tools.capture_3d_view"
        with patch(target) as mock:
            mock.return_value = _OK
            server.capture_3d_view(
                output_path="/tmp/v.png",
                width=800,
                height=600,
                view_index=1,
            )
            mock.assert_called_once_with("/tmp/v.png", 800, 600, 1)

    def test_workflow_modic_eval_all_args(self):
        target = "slicer_mcp.server.workflow_modic.workflow_modic_eval"
        with patch(target) as mock:
            mock.return_value = _OK
            server.workflow_modic_eval(
                t1_volume_id="t1",
                t2_volume_id="t2",
                region="cervical",
                segmentation_node_id="seg1",
                include_cord_screening=False,
            )
            mock.assert_called_once_with(
                t1_volume_id="t1",
                t2_volume_id="t2",
                region="cervical",
                segmentation_node_id="seg1",
                include_cord_screening=False,
            )

    def test_detect_cord_compression_mri_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_mri.detect_cord_compression_mri"
        with patch(target) as mock:
            mock.return_value = _OK
            server.detect_cord_compression_mri(
                t2_node_id="t2",
                t1_node_id="t1",
                region="thoracic",
                segmentation_node_id="seg1",
            )
            mock.assert_called_once_with("t2", "t1", "thoracic", "seg1")

    def test_detect_metastatic_lesions_ct_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_ct.detect_metastatic_lesions_ct"
        with patch(target) as mock:
            mock.return_value = _OK
            server.detect_metastatic_lesions_ct(
                volume_node_id="v1",
                segmentation_node_id="seg1",
                region="thoracic",
                include_posterior_elements=False,
            )
            mock.assert_called_once_with("v1", "seg1", "thoracic", False)

    def test_visualize_spine_segmentation_all_args(self):
        target = "slicer_mcp.server.spine_tools.visualize_spine_segmentation"
        with patch(target) as mock:
            mock.return_value = _OK
            server.visualize_spine_segmentation(
                segmentation_node_id="seg1",
                volume_node_id="v1",
                output_path="/out.png",
                region="full",
            )
            mock.assert_called_once_with("seg1", "v1", "/out.png", "full")

    def test_segment_vertebral_artery_all_args(self):
        target = "slicer_mcp.server.spine_tools.segment_vertebral_artery"
        with patch(target) as mock:
            mock.return_value = _OK
            server.segment_vertebral_artery(
                input_node_id="v1",
                side="left",
                seed_points=[[1.0, 2.0, 3.0]],
            )
            mock.assert_called_once_with("v1", "left", [[1.0, 2.0, 3.0]])

    def test_classify_disc_degeneration_xray_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_xray.classify_disc_degeneration_xray"
        with patch(target) as mock:
            mock.return_value = _OK
            server.classify_disc_degeneration_xray(
                volume_node_id="v1",
                landmarks_per_disc={},
                reference_disc_height_mm=8.5,
                magnification_factor=1.15,
            )
            mock.assert_called_once_with("v1", {}, 8.5, 1.15)

    def test_measure_listhesis_dynamic_xray_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_xray.measure_listhesis_dynamic_xray"
        with patch(target) as mock:
            mock.return_value = _OK
            server.measure_listhesis_dynamic_xray(
                volume_node_ids={"neutral": "n"},
                landmarks_per_position={},
                levels=["L4-L5"],
                region="cervical",
                magnification_factor=1.1,
            )
            mock.assert_called_once_with({"neutral": "n"}, {}, ["L4-L5"], "cervical", 1.1)

    def test_detect_metastatic_lesions_mri_all_args(self):
        target = "slicer_mcp.server.diagnostic_tools_mri.detect_metastatic_lesions_mri"
        with patch(target) as mock:
            mock.return_value = _OK
            server.detect_metastatic_lesions_mri(
                t1_node_id="t1",
                t2_stir_node_id="stir",
                region="thoracic",
                segmentation_node_id="seg1",
            )
            mock.assert_called_once_with("t1", "stir", "thoracic", "seg1")

    def test_place_landmarks_with_labels(self):
        target = "slicer_mcp.server.registration_tools.place_landmarks"
        with patch(target) as mock:
            mock.return_value = _OK
            server.place_landmarks(
                name="Pts",
                points=[[1, 2, 3], [4, 5, 6]],
                labels=["A", "B"],
            )
            mock.assert_called_once_with("Pts", [[1, 2, 3], [4, 5, 6]], ["A", "B"])
