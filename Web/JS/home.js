// =====================================================================
// 1. PASSWORD CHECKER LOGIC
// =====================================================================
const input = document.getElementById('pwdInput');
const strengthSection = document.getElementById('strengthSection');
const strengthBar = document.getElementById('strengthBar');
const clearBtn = document.getElementById('clearBtn');
const toggleVisBtn = document.getElementById('toggleVisBtn');
const reqUpper = document.getElementById('req-upper');
const reqDigit = document.getElementById('req-digit');
const reqSymbol = document.getElementById('req-symbol');
const time = document.getElementById('time');
const crackInfo = document.getElementById('crackInfo');

const regexUpper = /[A-Z]/;
const regexDigit = /[0-9]/;
const regexSymbol = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+/;
const strengthColors = ['#ff4d4d', '#ff944d', '#ffd24d', '#a3ff4d', '#00c49a'];

input.addEventListener('focus', () => strengthSection.classList.add('show'));
input.addEventListener('blur', () => {
    setTimeout(() => { if (document.activeElement !== input) strengthSection.classList.remove('show'); }, 100);
});

clearBtn.addEventListener('click', () => {
    input.value = '';
    input.dispatchEvent(new Event('input'));
    input.focus();
});

toggleVisBtn.addEventListener('click', () => {
    input.type = input.type === 'password' ? 'text' : 'password';
    toggleVisBtn.textContent = input.type === 'password' ? '🙈' : '🙉';
    input.focus();
});

async function checkPasswordLeak(password) {
    const reqLeak = document.getElementById('req-leak');
    if (!password) {
        reqLeak.classList.add('valid');
        reqLeak.innerHTML = '<div class="checkbox">✓</div> No leaks found';
        reqLeak.style.color = '';
        return;
    }

    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-1', data);
    const hashHex = Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('').toUpperCase();

    const prefix = hashHex.substring(0, 5);
    const suffix = hashHex.substring(5);

    try {
        const response = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`);
        const text = await response.text();
        const match = text.split('\n').find(line => line.startsWith(suffix));

        if (match) {
            const leakCount = match.split(':')[1].trim();
            reqLeak.classList.remove('valid');
            reqLeak.innerHTML = `<div class="checkbox" style="color: #ff4d4d; border-color: #ff4d4d;">✕</div> Leaked ${Number(leakCount).toLocaleString()} times!`;
            reqLeak.style.color = '#ff4d4d';
        } else {
            reqLeak.classList.add('valid');
            reqLeak.innerHTML = '<div class="checkbox">✓</div> No leaks found';
            reqLeak.style.color = '';
        }
    } catch (error) {
        console.error("Leak check error:", error);
    }
}

let leakCheckTimeout = null;

input.addEventListener('input', () => {
    const val = input.value;
    reqUpper.classList.toggle('valid', regexUpper.test(val));
    reqDigit.classList.toggle('valid', regexDigit.test(val));
    reqSymbol.classList.toggle('valid', regexSymbol.test(val));

    if (val === '') {
        strengthBar.style.width = '0%';
        strengthBar.style.backgroundColor = 'transparent';
        time.style.display = 'none';
        crackInfo.style.display = 'none';
        checkPasswordLeak('');
        return;
    }

    const result = zxcvbn(val);
    strengthBar.style.width = `${(result.score === 0 ? 20 : (result.score + 1) * 20)}%`;
    strengthBar.style.backgroundColor = strengthColors[result.score];
    time.style.display = 'block';
    time.textContent = result.crack_times_display.offline_slow_hashing_1e4_per_second;
    crackInfo.style.display = 'block';

    clearTimeout(leakCheckTimeout);
    leakCheckTimeout = setTimeout(() => checkPasswordLeak(val), 500);
});

// =====================================================================
// 2. AI GENERATOR LOGIC
// =====================================================================

const aiLengthSlider = document.getElementById('ai-gen-length');
const aiLengthVal = document.getElementById('ai-length-val');
const btnGenerate = document.getElementById('btn-generate');
const genResult = document.getElementById('gen-result');
const copyBtn = document.getElementById('copy-btn');

// Makes the AI slider number update when dragged
if (aiLengthSlider && aiLengthVal) {
    aiLengthSlider.addEventListener('input', () => {
        aiLengthVal.textContent = aiLengthSlider.value;
    });
}

// Copy button logic
if (copyBtn && genResult) {
    copyBtn.addEventListener('click', () => {
        if (genResult.value && genResult.value !== "AI is thinking..." && !genResult.value.includes("Error")) {
            genResult.select();
            document.execCommand('copy');
            const originalIcon = copyBtn.textContent;
            copyBtn.textContent = '✅';
            setTimeout(() => copyBtn.textContent = originalIcon, 1500);
        }
    });
}

// API Generate call
if (btnGenerate) {
    btnGenerate.addEventListener('click', async () => {
        let baseWord = document.getElementById('gen-base-word').value.trim();
        genResult.value = "AI is thinking...";

        // Grab values from the AI-specific inputs!
        const aiLength = parseInt(document.getElementById('ai-gen-length').value) || 16;
        const aiUpper = document.getElementById('ai-inc-upper').checked;
        const aiNum = document.getElementById('ai-inc-num').checked;
        const aiSym = document.getElementById('ai-inc-sym').checked;
        const aiAmbig = document.getElementById('ai-inc-ambig').checked;

        try {
            const response = await fetch('https://trustmiibro-backend.onrender.com/enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    weak_password: baseWord,
                    target_len: aiLength,
                    inc_upper: aiUpper,
                    inc_num: aiNum,
                    inc_sym: aiSym,
                    inc_ambig: aiAmbig 
                })
            });

            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            genResult.value = data.enhanced_password;

        } catch (error) {
            console.error("API Error:", error);
            genResult.value = "Error connecting to backend API.";
        }
    });
}