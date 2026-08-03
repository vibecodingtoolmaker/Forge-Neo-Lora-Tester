// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 vibecodingtoolmaker

(() => {
    "use strict";

    const DATAFRAME_IDS = [
        "txt2img_lora_tester_settings",
        "img2img_lora_tester_settings",
    ];
    const EDITOR_SELECTOR = 'input[role="textbox"]';
    const EDIT_PENDING_ATTRIBUTE = "data-lora-tester-edit-pending";

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
        // Column zero contains the LoRA path and is explicitly display-only.
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

    function handleCellFocus(event) {
        const target = eventElement(event);
        if (!target || target.closest(EDITOR_SELECTOR)) {
            return;
        }

        const cell = bodyCell(target);
        if (isEditableCell(cell)) {
            requestAnimationFrame(() => beginEditing(cell));
        }
    }

    function protectReadOnlyColumn(event) {
        const target = eventElement(event);
        const cell = bodyCell(target);
        if (cell?.cellIndex === 0) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
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
            cell.title = "LoRA identifier (read only)";
        });
    }

    function installDataframeBehavior() {
        for (const id of DATAFRAME_IDS) {
            const root = gradioApp().querySelector(`#${id}`);
            if (!root) {
                continue;
            }

            markReadOnlyCells(root);
            if (root.dataset.loraTesterCaretInstalled === "true") {
                continue;
            }

            root.dataset.loraTesterCaretInstalled = "true";
            root.addEventListener("click", handleCellClick);
            root.addEventListener("focusin", handleCellFocus);
            root.addEventListener("dblclick", protectReadOnlyColumn, true);
            root.addEventListener("keydown", handleTableKeydown, true);
        }
    }

    onUiLoaded(installDataframeBehavior);
    onAfterUiUpdate(installDataframeBehavior);
})();
