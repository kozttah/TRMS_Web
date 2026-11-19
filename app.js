let pyodide;
let editor;

// Estado global da simulação e animação 3D
let simData = null;
const FPS_INTERVAL = 1000 / 30; // 33.33 ms (30 FPS)

let animState = {
    playing: false,
    currentTime: 0,
    lastWallTime: 0,
    lastRenderTime: 0,
    Ts: 0.05
};

// Objetos globais do Three.js
let scene, camera, renderer, orbitControls;
let yawGroup, pitchGroup, mainPropeller, tailPropeller;

// 1. Motor de Simulação Python Otimizado para até 120s
const engineCode = `
import math
import numpy as np

def trms_dynamics_scalar(x_state, u0, u1, parafuso_apertado=False, u_pert=0.0):
    theta, theta_dot, psi, psi_dot, omega_p, omega_c, x_cross = x_state
    
    k1, T11, T10 = 1.1, 1.1, 1.0
    k2, T21, T20 = 0.8, 1.0, 1.0
    a1, b1 = 0.0135, 0.0924
    a2, b2 = 0.02, 0.09
    Mg = 0.32
    I1, I2 = 6.8e-2, 2.0e-2
    b1arfagem, b2arfagem = 6e-3, 1e-3
    b1guinada, b2guinada = 1e-1, 1e-2
    Kgy = 0.05
    Tp, T0, kc = 2.0, 3.5, -0.2

    Mprop_arfagem = a1 * (omega_p * omega_p) + b1 * omega_p
    Mprop_guinada = a2 * (omega_c * omega_c) + b2 * omega_c

    omega_p_dot = (1.0 / T11) * (-T10 * omega_p + k1 * u0)
    omega_c_dot = (1.0 / T21) * (-T20 * omega_c + k2 * u1)

    sgn_theta = 1.0 if theta_dot > 0.0 else (-1.0 if theta_dot < 0.0 else 0.0)
    sgn_psi   = 1.0 if psi_dot > 0.0 else (-1.0 if psi_dot < 0.0 else 0.0)

    Matrito_arfagem = b1arfagem * theta_dot + b2arfagem * sgn_theta
    Matrito_guinada = b1guinada * psi_dot + b2guinada * sgn_psi

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    Mpert = (u_pert * 9.81) * cos_theta * 0.5

    if parafuso_apertado:
        Mgiro = 0.0
        psi_ddot = 0.0
        x_cross_dot = 0.0
    else:
        Mgiro = Kgy * Mprop_guinada * psi_dot * cos_theta
        psi_ddot = (1.0 / I2) * (Mprop_guinada - Matrito_guinada - x_cross)
        x_cross_dot = (1.0 / Tp) * (-x_cross + kc * (T0 * omega_p_dot + omega_p))

    theta_ddot = (1.0 / I1) * (Mprop_arfagem - Mg * sin_theta - Matrito_arfagem - Mgiro + Mpert)

    return (theta_dot, theta_ddot, psi_dot, psi_ddot, omega_p_dot, omega_c_dot, x_cross_dot)

def run_simulation(user_code, t_end=20.0, Ts=0.05):
    local_env = {}
    try:
        exec(user_code, globals(), local_env)
        control_law = local_env['control_law']
    except Exception as e:
        return {"error": f"Erro de compilação: {str(e)}"}
        
    steps = int(t_end / Ts)
    t_data = np.zeros(steps)
    x_data = np.zeros((steps, 7))
    u_data = np.zeros((steps, 2))
    y_data = np.zeros((steps, 2))
    
    x = [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    dt_sim = 0.005
    half_dt = 0.5 * dt_sim
    sixth_dt = dt_sim / 6.0
    sim_steps = int(Ts / dt_sim)
    
    thetamin, thetamax = -0.8 * math.pi / 2.0, 0.8 * math.pi / 2.0
    psimin, psimax = -0.8 * math.pi, 0.8 * math.pi
    res_enc = 3.068e-3
    despad_ruido = 0.001
    
    noise = np.random.randn(steps, 2) * despad_ruido
    
    for k in range(steps):
        t_data[k] = k * Ts
        x_data[k, :] = x
        
        thetam = math.floor((x[0] + noise[k, 0]) / res_enc) * res_enc
        thetam = max(thetamin, min(thetamax, thetam))
        
        psim = math.floor((x[2] + noise[k, 1]) / res_enc) * res_enc
        psim = max(psimin, min(psimax, psim))
        
        y_data[k, 0] = thetam
        y_data[k, 1] = psim
        
        x_meas = np.array([thetam, x[1], psim, x[3], x[4], x[5], x[6]])
        
        try:
            u = control_law(k, x_meas, Ts)
            u0 = max(-2.5, min(2.5, float(u[0])))
            u1 = max(-2.5, min(2.5, float(u[1])))
        except Exception as e:
            return {"error": f"Erro na execução (k={k}): {str(e)}"}
            
        u_data[k, 0] = u0
        u_data[k, 1] = u1
        
        for _ in range(sim_steps):
            k1 = trms_dynamics_scalar(x, u0, u1)
            x2 = [x[i] + half_dt * k1[i] for i in range(7)]
            k2 = trms_dynamics_scalar(x2, u0, u1)
            x3 = [x[i] + half_dt * k2[i] for i in range(7)]
            k3 = trms_dynamics_scalar(x3, u0, u1)
            x4 = [x[i] + dt_sim * k3[i] for i in range(7)]
            k4 = trms_dynamics_scalar(x4, u0, u1)
            
            x = [x[i] + sixth_dt * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) for i in range(7)]
            
    return {
        "error": None,
        "t": t_data.tolist(),
        "pitch": y_data[:, 0].tolist(),
        "yaw": y_data[:, 1].tolist(),
        "pitch_real": x_data[:, 0].tolist(),
        "yaw_real": x_data[:, 2].tolist(),
        "omega_p": x_data[:, 4].tolist(),
        "omega_c": x_data[:, 5].tolist(),
        "u_p": u_data[:, 0].tolist(),
        "u_y": u_data[:, 1].tolist()
    }
`;

const defaultUserCode = `def control_law(k, x, Ts):
    """
    Projete sua lei de controle aqui.
    x[0]: arfagem (pitch) medida [rad]
    x[2]: guinada (yaw) medida [rad]
    A tensão máxima é +/- 2.5V.
    """
    ref_pitch = 0.2
    ref_yaw = 0.0
    
    kp_p, kd_p = 3.0, 2.0
    u_pitch = -kp_p * (x[0] - ref_pitch) - kd_p * x[1]
    
    kp_y, kd_y = 1.0, 0.8
    u_yaw = -kp_y * (x[2] - ref_yaw) - kd_y * x[3]
    
    return [u_pitch, u_yaw]
`;

async function initialize() {
    initThreeJS();

    const savedCode = localStorage.getItem('trms_user_code') || defaultUserCode;
    
    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.38.0/min/vs' }});
    require(['vs/editor/editor.main'], function() {
        editor = monaco.editor.create(document.getElementById('editor-container'), {
            value: savedCode,
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            minimap: { enabled: false },
            fontSize: 14,
            scrollBeyondLastLine: false
        });

        let timeout;
        editor.onDidChangeModelContent(() => {
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                localStorage.setItem('trms_user_code', editor.getValue());
            }, 1000);
        });
    });

    initEmptyPlots();

    const timeSlider = document.getElementById("time-slider");
    const timeVal = document.getElementById("time-val");
    timeSlider.addEventListener("input", (e) => {
        timeVal.textContent = e.target.value;
    });

    document.getElementById("btn-simulate").addEventListener("click", runSimulation);
    document.getElementById("btn-play").addEventListener("click", togglePlayAnimation);
    document.getElementById("btn-fullscreen").addEventListener("click", toggleFullScreen);

    pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    await pyodide.loadPackage("scipy");
    await pyodide.runPythonAsync(engineCode);

    document.getElementById("status").textContent = "Pronto (v3 - Tempo Real)";
    document.getElementById("status").style.color = "#4caf50";
    document.getElementById("btn-simulate").disabled = false;
}

// ==========================================
// CONSTRUÇÃO E RENDERIZAÇÃO 3D (THREE.JS A 30 FPS)
// ==========================================
function initThreeJS() {
    const container = document.getElementById('viewer-3d');
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x181a1f);

    camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(2.3, 2.1, 2.8);

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(1); // Resolução nativa 1:1 para leveza máxima no Safari
    container.appendChild(renderer.domElement);

    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.target.set(0, 1.25, 0);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.08;
    orbitControls.maxPolarAngle = Math.PI / 2 - 0.02;
    orbitControls.update();

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x222233, 0.75);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(4, 7, 5);
    scene.add(dirLight);

    const floorGeo = new THREE.PlaneGeometry(14, 14);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x141619, roughness: 0.9, metalness: 0.1 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const grid = new THREE.GridHelper(10, 20, 0x3a404d, 0x262930);
    grid.position.y = 0.002;
    scene.add(grid);

    const matDarkMetal = new THREE.MeshStandardMaterial({ color: 0x2b2d30, metalness: 0.7, roughness: 0.35 });
    const matAluminum = new THREE.MeshStandardMaterial({ color: 0xcfd6df, metalness: 0.85, roughness: 0.2 });
    const matBlue = new THREE.MeshStandardMaterial({ color: 0x0e639c, metalness: 0.4, roughness: 0.3 });
    const matRed = new THREE.MeshStandardMaterial({ color: 0xd63031, metalness: 0.3, roughness: 0.4 });
    const matBrass = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.25 });
    const matBlade = new THREE.MeshStandardMaterial({ color: 0xdfe6e9, metalness: 0.2, roughness: 0.4 });

    const basePlate = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.08, 0.9), matDarkMetal);
    basePlate.position.y = 0.04;
    scene.add(basePlate);

    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 1.35, 20), matDarkMetal);
    pedestal.position.y = 0.72;
    scene.add(pedestal);

    yawGroup = new THREE.Group();
    yawGroup.position.set(0, 1.42, 0);
    scene.add(yawGroup);

    const yawHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.16, 20), matBlue);
    yawGroup.add(yawHousing);

    pitchGroup = new THREE.Group();
    yawGroup.add(pitchGroup);

    const pitchHub = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.2, 16), matAluminum);
    pitchHub.rotation.x = Math.PI / 2;
    pitchGroup.add(pitchHub);

    const mainBeam = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1.7, 12), matAluminum);
    mainBeam.rotation.z = Math.PI / 2;
    pitchGroup.add(mainBeam);

    const cwRod = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.42, 10), matAluminum);
    cwRod.position.set(-0.08, -0.21, 0);
    cwRod.rotation.z = -0.15;
    pitchGroup.add(cwRod);

    const cwWeight = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.14, 16), matBrass);
    cwWeight.position.set(-0.11, -0.38, 0);
    cwWeight.rotation.z = -0.15;
    pitchGroup.add(cwWeight);

    const mainMotorGroup = new THREE.Group();
    mainMotorGroup.position.set(0.85, 0, 0);
    pitchGroup.add(mainMotorGroup);

    const mainMotorBody = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.18, 16), matDarkMetal);
    mainMotorGroup.add(mainMotorBody);

    const mainShroud = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.015, 10, 28), matRed);
    mainShroud.rotation.x = Math.PI / 2;
    mainShroud.position.y = 0.08;
    mainMotorGroup.add(mainShroud);

    mainPropeller = new THREE.Group();
    mainPropeller.position.y = 0.09;
    mainMotorGroup.add(mainPropeller);

    const mainBlade1 = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.008, 0.045), matBlade);
    const mainBlade2 = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.008, 0.48), matBlade);
    const mainSpinner = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.05, 12), matRed);
    mainSpinner.position.y = 0.02;
    mainPropeller.add(mainBlade1, mainBlade2, mainSpinner);

    const tailMotorGroup = new THREE.Group();
    tailMotorGroup.position.set(-0.85, 0, 0);
    pitchGroup.add(tailMotorGroup);

    const tailMotorBody = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.16, 16), matDarkMetal);
    tailMotorBody.rotation.x = Math.PI / 2;
    tailMotorGroup.add(tailMotorBody);

    const tailShroud = new THREE.Mesh(new THREE.TorusGeometry(0.20, 0.013, 10, 24), matBlue);
    tailShroud.position.z = 0.07;
    tailMotorGroup.add(tailShroud);

    tailPropeller = new THREE.Group();
    tailPropeller.position.z = 0.08;
    tailMotorGroup.add(tailPropeller);

    const tailBlade1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.035, 0.007), matBlade);
    const tailBlade2 = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.34, 0.007), matBlade);
    const tailSpinner = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.04, 12), matBlue);
    tailSpinner.rotation.x = Math.PI / 2;
    tailSpinner.position.z = 0.015;
    tailPropeller.add(tailBlade1, tailBlade2, tailSpinner);

    setTRMSPose(-0.3, 0.0);

    document.getElementById('btn-reset-cam').addEventListener('click', () => {
        camera.position.set(2.3, 2.1, 2.8);
        orbitControls.target.set(0, 1.25, 0);
        orbitControls.update();
    });

    document.getElementById('anim-speed').addEventListener('change', (e) => {
        const spdTxt = document.getElementById('hud-speed');
        if (spdTxt) spdTxt.textContent = `${e.target.value}x`;
    });

    const scrubber = document.getElementById('anim-scrubber');
    scrubber.addEventListener('input', (e) => {
        if (!simData) return;
        animState.playing = false;
        document.getElementById('btn-play').textContent = '🎬 Play 3D';
        const ratio = parseFloat(e.target.value) / 100;
        const totalDuration = simData.t[simData.t.length - 1];
        animState.currentTime = ratio * totalDuration;
        applyInterpolatedState(animState.currentTime, 0.033, 1.0);
        renderer.render(scene, camera);
    });

    window.addEventListener('resize', onWindowResize);
    animState.lastWallTime = performance.now();
    requestAnimationFrame(animateLoop);
}

function setTRMSPose(theta, psi) {
    if (pitchGroup && yawGroup) {
        pitchGroup.rotation.z = theta;
        yawGroup.rotation.y = psi;
    }
}

function applyInterpolatedState(tCurr, dtSimStep, speed) {
    if (!simData || simData.t.length === 0) return;

    const totalT = simData.t[simData.t.length - 1];
    const clampedT = Math.max(0, Math.min(tCurr, totalT));
    
    const exactIdx = clampedT / animState.Ts;
    const idx0 = Math.min(Math.floor(exactIdx), simData.t.length - 1);
    const idx1 = Math.min(idx0 + 1, simData.t.length - 1);
    const alpha = exactIdx - idx0;

    const theta = simData.pitch_real[idx0] * (1 - alpha) + simData.pitch_real[idx1] * alpha;
    const psi   = simData.yaw_real[idx0]   * (1 - alpha) + simData.yaw_real[idx1]   * alpha;
    const wp    = simData.omega_p[idx0]    * (1 - alpha) + simData.omega_p[idx1]    * alpha;
    const wc    = simData.omega_c[idx0]    * (1 - alpha) + simData.omega_c[idx1]    * alpha;

    setTRMSPose(theta, psi);

    mainPropeller.rotation.y += wp * dtSimStep * 20.0;
    tailPropeller.rotation.z += wc * dtSimStep * 20.0;

    document.getElementById('hud-t').textContent = clampedT.toFixed(2);
    const spdEl = document.getElementById('hud-speed');
    if (spdEl) spdEl.textContent = `${speed}x`;
    document.getElementById('hud-pitch').textContent = theta.toFixed(3);
    document.getElementById('hud-yaw').textContent = psi.toFixed(3);
    document.getElementById('anim-time-label').textContent = `${clampedT.toFixed(1)}s / ${totalT.toFixed(1)}s`;
    document.getElementById('anim-scrubber').value = ((clampedT / totalT) * 100).toFixed(1);
}

function animateLoop() {
    requestAnimationFrame(animateLoop);

    const now = performance.now();
    const dtReal = Math.min((now - animState.lastWallTime) / 1000.0, 0.5);
    animState.lastWallTime = now;

    if (animState.playing && simData) {
        const speedEl = document.getElementById('anim-speed');
        const speed = speedEl ? (parseFloat(speedEl.value) || 1.0) : 1.0;
        const totalT = simData.t[simData.t.length - 1];

        // Avança o tempo de simulação pelo tempo físico real multiplicado pelo seletor (1x, 2x, 4x, 8x)
        animState.currentTime += dtReal * speed;

        if (animState.currentTime >= totalT) {
            animState.currentTime = totalT;
            animState.playing = false;
            document.getElementById('btn-play').textContent = '🎬 Replay 3D';
        }
        applyInterpolatedState(animState.currentTime, dtReal * speed, speed);
    }

    // Renderiza a cena WebGL a 30 FPS
    if (now - animState.lastRenderTime >= FPS_INTERVAL) {
        animState.lastRenderTime = now;
        if (orbitControls) orbitControls.update();
        if (renderer && scene && camera) renderer.render(scene, camera);
    }
}

function togglePlayAnimation() {
    if (!simData) return;
    const btnPlay = document.getElementById('btn-play');
    const totalT = simData.t[simData.t.length - 1];

    if (animState.playing) {
        animState.playing = false;
        btnPlay.textContent = '🎬 Play 3D';
    } else {
        if (animState.currentTime >= totalT) {
            animState.currentTime = 0;
        }
        animState.lastWallTime = performance.now();
        animState.playing = true;
        btnPlay.textContent = '⏸ Pausar 3D';
    }
}

function onWindowResize() {
    const container = document.getElementById('viewer-3d');
    if (!container || !renderer || !camera) return;
    if (container.clientWidth === 0 || container.clientHeight === 0) return;

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// ==========================================
// EXECUÇÃO DO PYODIDE E PLOTLY
// ==========================================
async function runSimulation() {
    const btnSimulate = document.getElementById("btn-simulate");
    const btnPlay = document.getElementById("btn-play");
    const statusEl = document.getElementById("status");
    
    animState.playing = false;
    btnSimulate.disabled = true;
    btnPlay.disabled = true;
    statusEl.textContent = "Integrando modelo 2-DoF (RK4)...";
    statusEl.style.color = "#d7ba7d";
    
    const code = editor.getValue();
    localStorage.setItem('trms_user_code', code);
    
    try {
        const tEnd = parseFloat(document.getElementById("time-slider").value);
        
        pyodide.globals.set("student_code", code);
        pyodide.globals.set("t_end_js", tEnd);
        
        const pyResult = await pyodide.runPythonAsync('run_simulation(student_code, t_end=t_end_js)');
        const resultJS = pyResult.toJs();
        
        const errorMsg = resultJS.get("error");
        if (errorMsg) {
            alert(errorMsg);
            throw new Error(errorMsg);
        }

        simData = {
            t: resultJS.get("t"),
            pitch: resultJS.get("pitch"),
            yaw: resultJS.get("yaw"),
            pitch_real: resultJS.get("pitch_real"),
            yaw_real: resultJS.get("yaw_real"),
            omega_p: resultJS.get("omega_p"),
            omega_c: resultJS.get("omega_c"),
            u_p: resultJS.get("u_p"),
            u_y: resultJS.get("u_y")
        };

        updatePlots(simData.t, simData.pitch, simData.yaw, simData.u_p, simData.u_y);
        
        document.getElementById("anim-scrubber").disabled = false;
        btnPlay.disabled = false;
        animState.currentTime = 0;
        animState.lastWallTime = performance.now();
        animState.playing = true;
        btnPlay.textContent = '⏸ Pausar 3D';
        
    } catch (err) {
        console.error("Erro no Pyodide:", err);
    } finally {
        btnSimulate.disabled = false;
        statusEl.textContent = "Simulação concluída (v3).";
        statusEl.style.color = "#4caf50";
    }
}

const darkLayout = {
    plot_bgcolor: '#252526',
    paper_bgcolor: '#252526',
    font: { color: '#cccccc', size: 11 },
    margin: { t: 32, b: 32, l: 48, r: 48 },
    xaxis: { gridcolor: '#3c3c3c', title: 'Tempo (s)' },
    yaxis: { gridcolor: '#3c3c3c', title: 'Ângulo (rad)' },
    yaxis2: { overlaying: 'y', side: 'right', showgrid: false, title: 'Ação (V)' },
    showlegend: true,
    legend: { x: 0, y: 1.15, orientation: 'h' }
};

function initEmptyPlots() {
    Plotly.newPlot('plot-pitch', [], { ...darkLayout, title: 'Pitch (Arfagem) - Quantizado' }, {responsive: true});
    Plotly.newPlot('plot-yaw', [], { ...darkLayout, title: 'Yaw (Guinada) - Quantizado' }, {responsive: true});
}

function updatePlots(t, pitch, yaw, u_p, u_y) {
    const tracePitch = { x: t, y: pitch, name: 'Pitch', line: {color: '#e74c3c'} };
    const traceUp = { x: t, y: u_p, name: 'u_p (V)', yaxis: 'y2', line: {color: '#e74c3c', dash: 'dot', width: 1, shape: 'hv'} };
    Plotly.react('plot-pitch', [tracePitch, traceUp], { ...darkLayout, title: 'Pitch (Arfagem)' });

    const traceYaw = { x: t, y: yaw, name: 'Yaw', line: {color: '#3498db'} };
    const traceUy = { x: t, y: u_y, name: 'u_y (V)', yaxis: 'y2', line: {color: '#3498db', dash: 'dot', width: 1, shape: 'hv'} };
    Plotly.react('plot-yaw', [traceYaw, traceUy], { ...darkLayout, title: 'Yaw (Guinada)' });
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
    }
}

function switchTab(tabId, btnElement) {
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
    
    if (tabId === 'tab-sim') {
        onWindowResize();
        window.dispatchEvent(new Event('resize'));
    }
}

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}