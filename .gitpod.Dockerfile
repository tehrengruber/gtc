FROM gitpod/workspace-full

COPY .gitpod/aliases.txt .
RUN cat bashrc >> $HOME/.bashrc
