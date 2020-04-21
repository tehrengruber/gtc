FROM gitpod/workspace-full

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc
