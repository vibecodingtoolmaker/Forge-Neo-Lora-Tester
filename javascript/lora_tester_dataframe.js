// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 vibecodingtoolmaker

(() => {
    "use strict";

    const DATAFRAME_IDS = [
        "txt2img_lora_tester_settings",
        "img2img_lora_tester_settings",
        "txt2img_lora_tester_embedding_settings",
        "img2img_lora_tester_embedding_settings",
    ];
    const EDITOR_SELECTOR = 'input[role="textbox"]';
    const EDIT_PENDING_ATTRIBUTE = "data-lora-tester-edit-pending";
    const PRESET_BRIDGES = ["txt2img", "img2img"];
    const DIMENSION_BRIDGES = ["txt2img", "img2img"];
    const dimensionTimers = new Map();

    function eventElement(event) {
        return event.target instanceof Element ? event.target : null;
    }

    function bodyCell(element) {
        const cell = element?.closest("td");
        if (!cell || cell.tabIndex < 0 || !cell.closest("tbody")) {
            return null;
        }
        return cell;
    }

    function selectedBodyCell(root) {
        const activeElement = document.activeElement;
        const activeCell = activeElement instanceof Element
            ? bodyCell(activeElement)
            : null;
        if (activeCell && root.contains(activeCell)) {
            return activeCell;
        }

        return root.querySelector('tbody td.focus[tabindex="0"]');
    }

    function isEditableCell(cell) {
        // Column zero contains the LoRA/Embedding path and is display-only.
        return cell !== null && cell.cellIndex > 0;
    }

    function placeCaretAtStart(cell, remainingAttempts = 4) {
        const editor = cell.querySelector(EDITOR_SELECTOR);
        if (!editor) {
            if (remainingAttempts > 0) {
                requestAnimationFrame(() => {
                    placeCaretAtStart(cell, remainingAttempts - 1);
                });
            }
            return;
        }

        editor.focus({preventScroll: true});
        if (typeof editor.setSelectionRange === "function") {
            editor.setSelectionRange(0, 0);
        }
    }

    function beginEditing(cell) {
        if (!isEditableCell(cell)) {
            return;
        }

        const existingEditor = cell.querySelector(EDITOR_SELECTOR);
        if (existingEditor) {
            placeCaretAtStart(cell);
            return;
        }

        if (cell.hasAttribute(EDIT_PENDING_ATTRIBUTE)) {
            return;
        }

        cell.setAttribute(EDIT_PENDING_ATTRIBUTE, "true");
        cell.dispatchEvent(new MouseEvent("dblclick", {
            bubbles: true,
            cancelable: true,
            view: window,
        }));

        requestAnimationFrame(() => {
            cell.removeAttribute(EDIT_PENDING_ATTRIBUTE);
            placeCaretAtStart(cell);
        });
    }

    function handleCellClick(event) {
        const target = eventElement(event);
        if (!target || target.closest(EDITOR_SELECTOR)) {
            return;
        }

        const cell = bodyCell(target);
        if (!isEditableCell(cell)) {
            return;
        }

        // Gradio 4.40 uses the first click only for cell selection and a double
        // click for edit mode. Let its click handler finish, then enter edit mode.
        requestAnimationFrame(() => beginEditing(cell));
    }

    function protectReadOnlyColumn(event) {
        const target = eventElement(event);
        const cell = bodyCell(target);
        if (cell?.cellIndex === 0) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }

    function releaseEditorForScroll(event) {
        const root = event.currentTarget;
        const editor = document.activeElement;
        if (!(editor instanceof HTMLInputElement) || !root.contains(editor)) {
            return;
        }

        // Gradio's virtualized Dataframe keeps a focused editor inside the
        // viewport.  When the user scrolls, commit the current input state and
        // release focus before the virtualizer can pull the table back to it.
        editor.dispatchEvent(new Event("change", {bubbles: true}));
        editor.blur();
    }

    function handleTableKeydown(event) {
        const root = event.currentTarget;
        const cell = selectedBodyCell(root);

        if (cell?.cellIndex === 0) {
            const printable = event.key.length === 1
                && !event.ctrlKey
                && !event.metaKey
                && !event.altKey;
            const destructive = ["Backspace", "Delete", "Enter"].includes(event.key);
            if (printable || destructive) {
                event.preventDefault();
                event.stopImmediatePropagation();
                return;
            }
        }

        const target = eventElement(event);
        const editorActive = Boolean(target?.closest(EDITOR_SELECTOR));
        const movesSelectedCell = event.key === "Tab"
            || (!editorActive && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Enter"].includes(event.key));
        if (movesSelectedCell) {
            // Gradio changes the selected cell during the same key event. Enter
            // edit mode only after that move has completed.
            setTimeout(() => {
                beginEditing(selectedBodyCell(root));
            }, 0);
        }
    }

    function markReadOnlyCells(root) {
        root.querySelectorAll('tbody td:first-child[tabindex="0"]').forEach((cell) => {
            cell.setAttribute("aria-readonly", "true");
            cell.title = "Model item identifier (read only)";
        });
    }

    function hideDynamicBlankRows(root) {
        root.querySelectorAll("tbody tr").forEach((row) => {
            const text = row.textContent?.trim() || "";
            const editorHasValue = Array.from(
                row.querySelectorAll("input, textarea"),
            ).some((editor) => editor.value.trim() !== "");
            row.classList.toggle(
                "lora-tester-empty-dataframe-row",
                text === "" && !editorHasValue,
            );
        });
    }

    function hideRowInsertionControls(root) {
        root.querySelectorAll("button").forEach((button) => {
            const label = (button.textContent || button.getAttribute("aria-label") || "")
                .trim()
                .toLowerCase();
            if (label === "new row") {
                button.hidden = true;
            }
        });
    }

    function installDataframeBehavior() {
        for (const id of DATAFRAME_IDS) {
            const root = gradioApp().querySelector(`#${id}`);
            if (!root) {
                continue;
            }

            hideDynamicBlankRows(root);
            hideRowInsertionControls(root);
            markReadOnlyCells(root);
            if (root.dataset.loraTesterCaretInstalled === "true") {
                continue;
            }

            root.dataset.loraTesterCaretInstalled = "true";
            root.addEventListener("click", handleCellClick);
            root.addEventListener("dblclick", protectReadOnlyColumn, true);
            root.addEventListener("keydown", handleTableKeydown, true);
            root.addEventListener("wheel", releaseEditorForScroll, {
                capture: true,
                passive: true,
            });
        }
    }

    function setInputValue(input, value) {
        const prototype = input instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(
            prototype,
            "value",
        )?.set;
        if (setter) {
            setter.call(input, value);
        } else {
            input.value = value;
        }
        input.dispatchEvent(new Event("input", {bubbles: true}));
        input.dispatchEvent(new Event("change", {bubbles: true}));
    }

    function selectedForgePreset() {
        const root = gradioApp().querySelector("#forge_ui_preset");
        const input = root?.querySelector("input");
        return input?.value?.trim().toLowerCase() || null;
    }

    function syncForgePreset(force = false) {
        const preset = selectedForgePreset();
        if (!preset) {
            return;
        }

        for (const mode of PRESET_BRIDGES) {
            const bridgeRoot = gradioApp().querySelector(
                `#${mode}_lora_tester_preset_bridge`,
            );
            const bridge = bridgeRoot?.querySelector("textarea, input");
            const refresh = gradioApp().querySelector(
                `#${mode}_lora_tester_preset_refresh`,
            );
            if (!bridge || !refresh) {
                continue;
            }
            if (!force && bridge.value.trim().toLowerCase() === preset) {
                continue;
            }
            setInputValue(bridge, preset);
            setTimeout(() => refresh.click(), 0);
        }
    }

    function installPresetBridge() {
        const root = gradioApp().querySelector("#forge_ui_preset");
        if (!root) {
            return;
        }
        if (root.dataset.loraTesterPresetInstalled !== "true") {
            root.dataset.loraTesterPresetInstalled = "true";
            const scheduleSync = () => setTimeout(() => syncForgePreset(), 0);
            root.addEventListener("input", scheduleSync);
            root.addEventListener("change", scheduleSync);
        }
        syncForgePreset();
    }

    function syncComparisonDimensions(mode, force = false) {
        const width = gradioApp().querySelector(`#${mode}_width input`)?.value;
        const height = gradioApp().querySelector(`#${mode}_height input`)?.value;
        const widthBridge = gradioApp().querySelector(
            `#${mode}_lora_tester_comparison_width input`,
        );
        const heightBridge = gradioApp().querySelector(
            `#${mode}_lora_tester_comparison_height input`,
        );
        const refresh = gradioApp().querySelector(
            `#${mode}_lora_tester_comparison_refresh`,
        );
        if (!width || !height || !widthBridge || !heightBridge || !refresh) {
            return;
        }
        if (
            !force
            && widthBridge.value === width
            && heightBridge.value === height
        ) {
            return;
        }
        setInputValue(widthBridge, width);
        setInputValue(heightBridge, height);
        setTimeout(() => refresh.click(), 0);
    }

    function scheduleDimensionSync(mode) {
        clearTimeout(dimensionTimers.get(mode));
        dimensionTimers.set(
            mode,
            setTimeout(() => syncComparisonDimensions(mode), 150),
        );
    }

    function installDimensionBridges() {
        for (const mode of DIMENSION_BRIDGES) {
            for (const dimension of ["width", "height"]) {
                const root = gradioApp().querySelector(`#${mode}_${dimension}`);
                if (!root || root.dataset.loraTesterDimensionInstalled === "true") {
                    continue;
                }
                root.dataset.loraTesterDimensionInstalled = "true";
                const schedule = () => scheduleDimensionSync(mode);
                root.addEventListener("input", schedule);
                root.addEventListener("change", schedule);
            }
            syncComparisonDimensions(mode);
        }
    }

    onUiLoaded(() => {
        installDataframeBehavior();
        installPresetBridge();
        installDimensionBridges();
    });
    onAfterUiUpdate(() => {
        installDataframeBehavior();
        installPresetBridge();
        installDimensionBridges();
    });
})();
