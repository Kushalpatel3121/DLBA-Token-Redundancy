# Instructions: running the project on a VCL VM

How this setup works, in one picture:

```
 YOUR UBUNTU LAPTOP  (the permanent home of everything)         VCL VM  (temporary, wiped when the reservation ends)
 ~/dlba-project/   <------------- rsync push / pull ------------>   ~/dlba-project/
   code, configs, checkpoints, results, notes                         where the GPU work happens
   git repo  ---> GitHub  (code + small results only)                 venv at ~/venv-dlba
```

**Golden rules**
1. The laptop is the source of truth. The VM is disposable.
2. `pull` often. Always pull before a reservation ends. Anything only on the VM is lost.
3. Git is for code and small files. Big files (checkpoints, caches) travel only by rsync.
4. Run anything long inside **tmux**, so it survives a dropped connection or a closed VS Code.
5. Run the `vmctl.sh` commands on the **laptop**. Run the Python work on the **VM**.

---

## Part A. One-time laptop setup

### A1. Tools
```bash
sudo apt update
sudo apt install -y openssh-client rsync git unzip
```

### A2. VS Code and the Remote-SSH extension
VS Code runs on your laptop, but opens a window whose files and terminal are on the VM.

1. Install VS Code if you do not have it: download the `.deb` from code.visualstudio.com and run
   `sudo apt install ./<downloaded-file>.deb` (or `sudo snap install code --classic`).
2. Install the extension:
   ```bash
   code --install-extension ms-vscode-remote.remote-ssh
   ```
   (or: in VS Code press `Ctrl+Shift+X`, search "Remote - SSH", install the Microsoft one).

### A3. Put the project on the laptop
```bash
unzip dlba-project.zip -d ~/
cd ~/dlba-project
bash scripts/make_dirs.sh        # makes sure every folder exists
chmod +x scripts/*.sh
```

### A4. Git and GitHub (on the laptop only; the VM never touches GitHub)
1. Tell git who you are (once per laptop):
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```
2. Make an SSH key for GitHub and add it to your account:
   ```bash
   ssh-keygen -t ed25519 -C "you@example.com"      # press Enter to accept the defaults
   cat ~/.ssh/id_ed25519.pub                      # copy this whole line
   ```
   On github.com: Settings -> SSH and GPG keys -> New SSH key -> paste. Test with `ssh -T git@github.com`.
3. On github.com create a new **private** repository (empty: no README, no .gitignore).
4. Connect and push:
   ```bash
   cd ~/dlba-project
   git init -b main
   git add -A
   git commit -m "Initial project skeleton"
   git remote add origin git@github.com:<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Add your teammate as a collaborator on the repo (Settings -> Collaborators).

---

## Part B. Every new VM reservation (and the very first one)

This is also how you **rebuild everything on a new VM and continue where you stopped**.
Because the laptop holds the latest state, the same three commands work every time.

### B1. Get the connection details
When the VCL reservation is ready, it shows the VM's IP address (or hostname) and your username.
If you are off campus you may need the university VPN to reach it.

### B2. Connect the VM to your laptop
```bash
cd ~/dlba-project
./scripts/vmctl.sh init
```
It asks for the host and username, creates an SSH key for VCL (first time only), writes a `dlba-vm`
entry in `~/.ssh/config`, copies your key to the VM (you type your password once), and tests the
connection. A new reservation means a new IP, so run `init` again each time and enter the new IP.

### B3. Build the environment
First time ever:
```bash
./scripts/vmctl.sh setup --freeze
```
Every later VM:
```bash
./scripts/vmctl.sh setup
```
What this does, in order:
1. `push`: copies the whole project (code, configs, checkpoints, results, notes) to the VM.
2. Runs `scripts/bootstrap_vm.sh` on the VM, which: checks the machine and GPU, installs tmux, git,
   rsync and python3-venv if missing, creates the folders, creates the venv at `~/venv-dlba`,
   installs the packages (the **frozen** versions from `requirements.txt` if that file exists,
   otherwise `requirements.in`), writes a tmux config and a shortcut, then runs `verify_env.py`.
3. With `--freeze`, and only if verification passed, writes the exact versions to `requirements.txt`.

**Success looks like:** every line `[ OK ]`, and the summary says `0 failed`. A `[WARN]` on disk space
or precision flags is not fatal. If anything says `[FAIL]`, copy the full output and ask for help
before going further (see Part E for the common ones).

Then save the frozen file on the laptop and in git:
```bash
./scripts/vmctl.sh sync "freeze requirements"
```

### B4. Connect VS Code to the VM
1. Open VS Code on the laptop.
2. Press `Ctrl+Shift+P` and run **Remote-SSH: Connect to Host...**, then choose **dlba-vm**.
3. If it asks for the platform, pick **Linux**. A new window opens. The first connection takes a minute
   because VS Code installs a small server on the VM. The green box at the bottom-left should read `SSH: dlba-vm`.
4. **File -> Open Folder...** and open `/home/<your-username>/dlba-project` (on the VM).
5. Open the Extensions panel (`Ctrl+Shift+X`) and install **Python** and **Jupyter** **in SSH: dlba-vm**
   (the install button says so). You do this once per new VM.
6. `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose `~/venv-dlba/bin/python`.
7. Open a terminal inside VS Code (`` Ctrl+` ``). It is a shell on the VM.

### B5. Start tmux for anything long-running
In the VS Code terminal (or from the laptop with `./scripts/vmctl.sh shell`):
```bash
tmux new -s main        # or: tmux attach -t main   if it already exists
dlba                    # shortcut: activates the venv and goes to the project folder
```
Start long jobs (training, leave-one-out) **inside tmux**. See the cheat sheet in Part F.

---

## Part C. Working loop

1. Edit code in VS Code (it is saving directly on the VM).
2. Run jobs in tmux. Read logs with `tail -f logs/<file>`.
3. **Every 30 to 60 minutes**, and after every important result, on the laptop:
   ```bash
   ./scripts/vmctl.sh sync "short message about what changed"
   ```
   This pulls everything from the VM, commits the small files to git, and pushes to GitHub.
   (`pull` alone only copies files, without git.)
4. Check on the machine any time with `./scripts/vmctl.sh status`.

**Never edit the same file on the laptop and on the VM at the same time.** rsync keeps the newer
file, which is safe, but you could silently lose the older edit.

---

## Part D. End of every session (do not skip)

1. On the VM, make sure no job is still writing files (`tmux ls`, check the running windows).
2. Update `PROGRESS.md` with what is done, what is half-finished, and the **first thing to do next time**.
3. On the laptop:
   ```bash
   ./scripts/vmctl.sh pull         # final copy of everything
   ./scripts/vmctl.sh sync "end of session"
   ```
4. Do a dry run if you are unsure what moved: `DRY=1 ./scripts/vmctl.sh pull`.
5. Check that the checkpoints and results are really on the laptop before the reservation expires.
   Reservations are capped at 24 hours, and a VM that has expired cannot be recovered.

---

## Part E. Troubleshooting

| Problem | What to try |
|---|---|
| `init` cannot connect | Wrong IP or username, VM still starting, or you are off campus without the VPN. |
| "Host key verification failed" / "REMOTE HOST IDENTIFICATION HAS CHANGED" | Normal after a new reservation reuses an IP. `init` already runs `ssh-keygen -R` for the new host. Or run it by hand: `ssh-keygen -R <ip>`. |
| Asks for the password every time | The key was not installed. Run `ssh-copy-id -i ~/.ssh/id_ed25519_vcl.pub dlba-vm`. The config also reuses one connection for 4 hours, so you normally type it once. |
| VS Code stuck on "Setting up SSH Host" | The VM must reach the internet to download the VS Code server. Test `ssh dlba-vm 'curl -sI https://update.code.visualstudio.com | head -1'`. Also try `Ctrl+Shift+P` -> "Remote-SSH: Kill VS Code Server on Host", then reconnect. |
| `pip install` fails, mentions a `+cuXXX` torch version | The frozen file was made with a specific CUDA build of torch. Set `TORCH_INDEX` on the laptop, e.g. `TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/vmctl.sh setup` (choose the one matching the CUDA version in `nvidia-smi`). |
| verify says `torch.cuda.is_available()` is False | You may be on a non-GPU image, or the torch build does not match the driver. Check `nvidia-smi`; the "CUDA Version" at the top right is the highest CUDA the driver supports. |
| verify says attention weights not returned | The model is not using the eager attention backend. The code must load the model with `attn_implementation="eager"` (already set in `configs/config.yaml`). |
| verify: manual layer loop mismatch / prune_heads fails | Almost always a `transformers` version issue. Pin `transformers>=4.44,<5` (the default), reinstall, and rerun. Paste the output if it persists. |
| verify: cannot download MRPC or the model | The VM has no internet access to Hugging Face. Download on the laptop (it caches in `~/.cache/huggingface`) and copy to the VM's `~/.cache/huggingface`. |
| `sudo` asks for a password and the script hangs | Run the bootstrap in an interactive shell on the VM (`./scripts/vmctl.sh shell`, then `bash scripts/bootstrap_vm.sh`) so you can type it. |
| rsync says "command not found" on the VM | Run `sudo apt-get install -y rsync` on the VM, or just rerun `./scripts/vmctl.sh setup`. |
| I overwrote something by mistake | rsync never deletes, and skips files newer on the other side. Check git history on the laptop (`git log`, `git checkout <commit> -- <file>`) for code and small files. |

---

## Part F. tmux cheat sheet

All keys start with `Ctrl+b`, then release, then press the next key.

| Goal | Keys / command |
|---|---|
| New named session | `tmux new -s main` |
| Detach (jobs keep running) | `Ctrl+b` then `d` |
| Re-attach | `tmux attach -t main` |
| List sessions | `tmux ls` |
| New window / next / previous | `Ctrl+b c` / `Ctrl+b n` / `Ctrl+b p` |
| Scroll back | mouse wheel (mouse mode is on), or `Ctrl+b [` then arrow keys, `q` to exit |
| Split side by side / top-bottom | `Ctrl+b %` / `Ctrl+b "` |
| Kill the current window | type `exit` |

Run a training job so its output is also saved to a log:
```bash
python src/<script>.py 2>&1 | tee logs/<name>.log
```

---

## Part G. What is in the project

| Path | Purpose |
|---|---|
| `INSTRUCTIONS.md` | This file |
| `PROGRESS.md` | Checklist and session notes: **where we left off** |
| `requirements.in` | Loose package list (used until a frozen file exists) |
| `requirements.txt` | Frozen exact versions (created by `--freeze`; commit it) |
| `configs/config.yaml` | Every agreed setting: seeds, epochs, learning rates, epsilon, drop layers, paths |
| `scripts/vmctl.sh` | Laptop-side control: `init`, `setup`, `shell`, `push`, `pull`, `sync`, `status` |
| `scripts/bootstrap_vm.sh` | VM-side environment builder |
| `scripts/verify_env.py` | Environment test (GPU, attention weights, layer loop, pruning, data) |
| `scripts/make_dirs.sh` | Creates the folder structure |
| `src/` | Project code (to be written) |
| `data/splits/` | The saved validation split (create once, never change) |
| `checkpoints/{M0,M0plus,M25,M50}/` | Model checkpoints (not in git) |
| `cache/hidden_states/` | Cached layer outputs for leave-one-out (not in git) |
| `results/{labels,signals,metrics}/` | Experiment outputs |
| `logs/` | Run logs and `env_report.txt` |
| `figures/` | Plots for the presentation |
| `.vmenv` | Your current VM host/user (not in git; rewritten by `init`) |

### Settings that were fixed on purpose (see `configs/config.yaml`)
- Seeds 42, 123, 2026. fp32 everywhere. Eager attention (needed to read attention weights).
- Validation split (10% of train, seed 42, saved to a file) for all calibration; the official dev set only for final numbers.
- M0: lr 2e-5, batch 32, 3 epochs, 10% warmup, best epoch by validation.
- M0+, M25, M50: identical 2 extra epochs, lr 1e-5, linear decay, final epoch kept.
- Pruned-vs-unpruned comparisons always use **M0+** as the reference.
- Redundancy threshold epsilon 0.01 (sensitivity 0.005 and 0.02), drop layers 4, 6, 8.
