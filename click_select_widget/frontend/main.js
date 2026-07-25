// Native <select> based Streamlit component.
// A real browser dropdown: click to open, click an option to choose.
// There is no text field, so typing/editing is not possible at all.

function sendValue(value) {
  Streamlit.setComponentValue(value);
}

let lastOptionsKey = null;

function onRender(event) {
  const { options, labels, index } = event.detail.args;
  const dropdown = document.getElementById("dropdown");

  const optionsKey = JSON.stringify(options) + "|" + JSON.stringify(labels);

  if (optionsKey !== lastOptionsKey) {
    dropdown.innerHTML = "";
    options.forEach((opt, i) => {
      const el = document.createElement("option");
      el.value = String(i);
      el.textContent = labels ? labels[i] : opt;
      dropdown.appendChild(el);
    });
    lastOptionsKey = optionsKey;
  }

  dropdown.value = String(index || 0);

  dropdown.onchange = () => {
    const i = parseInt(dropdown.value, 10);
    sendValue(options[i]);
  };

  Streamlit.setFrameHeight(48);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
