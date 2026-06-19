const presets = {
    balanced: {
        LIMIT_BAL: 50000, AGE: 35, SEX: 2, EDUCATION: 2, MARRIAGE: 2,
        PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
        BILL_AMT1: 18000, BILL_AMT2: 17500, BILL_AMT3: 16800, BILL_AMT4: 15000, BILL_AMT5: 14000, BILL_AMT6: 13000,
        PAY_AMT1: 1000, PAY_AMT2: 1200, PAY_AMT3: 1500, PAY_AMT4: 1000, PAY_AMT5: 1000, PAY_AMT6: 800,
    },
    low: {
        LIMIT_BAL: 200000, AGE: 40, SEX: 2, EDUCATION: 1, MARRIAGE: 2,
        PAY_0: -1, PAY_2: -1, PAY_3: -1, PAY_4: -1, PAY_5: -1, PAY_6: -1,
        BILL_AMT1: 30000, BILL_AMT2: 28000, BILL_AMT3: 25000, BILL_AMT4: 22000, BILL_AMT5: 20000, BILL_AMT6: 18000,
        PAY_AMT1: 15000, PAY_AMT2: 12000, PAY_AMT3: 10000, PAY_AMT4: 9000, PAY_AMT5: 8000, PAY_AMT6: 7000,
    },
    high: {
        LIMIT_BAL: 20000, AGE: 30, SEX: 1, EDUCATION: 2, MARRIAGE: 1,
        PAY_0: 3, PAY_2: 2, PAY_3: 2, PAY_4: 2, PAY_5: 1, PAY_6: 1,
        BILL_AMT1: 19500, BILL_AMT2: 19000, BILL_AMT3: 18500, BILL_AMT4: 18000, BILL_AMT5: 17500, BILL_AMT6: 17000,
        PAY_AMT1: 0, PAY_AMT2: 500, PAY_AMT3: 0, PAY_AMT4: 500, PAY_AMT5: 0, PAY_AMT6: 500,
    },
};

const form = document.querySelector("#risk-form");
const gaugeNumber = document.querySelector("#gauge-number");
const arcFill = document.querySelector("#arc-fill");
const thresholdEl = document.querySelector("#threshold");
const creditUsageEl = document.querySelector("#credit-usage");
const statusPill = document.querySelector("#status-pill");
const warningsEl = document.querySelector("#warnings");
const presetButtons = document.querySelectorAll(".preset");

const ARC_LENGTH = 283;
const buttonHtml = form.querySelector(".primary-button").innerHTML;

function setFormValues(values) {
    Object.entries(values).forEach(([name, value]) => {
        const element = form.elements[name];
        if (element) {
            element.value = value;
        }
    });
}

function formPayload() {
    const payload = {};

    for (const [key, value] of new FormData(form).entries()) {
        payload[key] = Number(value);
    }

    return payload;
}

function setGauge(probability, label) {
    const percent = Math.round(probability * 100);
    const offset = ARC_LENGTH - probability * ARC_LENGTH;
    const isHighRisk = label === "High risk";
    const isModerateRisk = label === "Moderate risk";

    arcFill.style.stroke = isHighRisk ? "url(#arcWarn)" : isModerateRisk ? "#f59e0b" : "url(#arcGrad)";
    arcFill.style.strokeDashoffset = offset;
    gaugeNumber.innerHTML = `${percent}<span>%</span>`;
}

function updateStatus(label) {
    statusPill.textContent = label;
    statusPill.className = "status-pill";

    if (label === "High risk") {
        statusPill.classList.add("high");
    } else if (label === "Moderate risk") {
        statusPill.classList.add("moderate");
    } else {
        statusPill.classList.add("low");
    }
}

function updateWarnings(warnings) {
    warningsEl.innerHTML = "";

    warnings.forEach((warning) => {
        const item = document.createElement("li");
        item.textContent = warning;
        warningsEl.appendChild(item);
    });
}

presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
        presetButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        setFormValues(presets[button.dataset.preset]);
    });
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const submitButton = form.querySelector(".primary-button");
    submitButton.textContent = "Analyzing...";
    submitButton.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formPayload()),
        });

        const result = await response.json();

        if (!response.ok) {
            alert(result.error || "Prediction failed.");
            return;
        }

        setGauge(result.probability, result.label);
        updateStatus(result.label);
        thresholdEl.textContent = `${result.threshold_percent}%`;
        creditUsageEl.textContent = `${result.credit_usage_percent}%`;
        updateWarnings(result.warnings);
    } finally {
        submitButton.innerHTML = buttonHtml;
        submitButton.disabled = false;
    }
});
