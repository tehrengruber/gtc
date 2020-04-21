FROM gitpod/workspace-full

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc

USER root

ARG ECBUILD_VERSION=3.3.2
RUN cd /tmp && \
    wget -q https://github.com/ecmwf/ecbuild/archive/${ECBUILD_VERSION}.tar.gz && \
    tar xzf ${ECBUILD_VERSION}.tar.gz && \
    mv ecbuild-${ECBUILD_VERSION} $HOME && \
    rm -rf /tmp/*

# ARG ECKIT_VERSION=1.10.1
# RUN export PATH=$PATH:$HOME/ecbuild-${ECBUILD_VERSION}/bin && \
#     cd /tmp && \
#     wget -q https://github.com/ecmwf/eckit/archive/${ECKIT_VERSION}.tar.gz && \
#     tar xzf ${ECKIT_VERSION}.tar.gz && \
#     cd eckit-${ECKIT_VERSION} && \
#     mkdir build && cd build && \
#     cmake .. && \
#     make -j$(nproc) install && \
#     rm -rf /tmp/*


# ARG ATLAS_VERSION=0.20.1
# RUN cd /tmp && \
#     wget -q https://github.com/ecmwf/atlas/archive/${ATLAS_VERSION}.tar.gz && \

USER gitpod
