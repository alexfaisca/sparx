#!/usr/bin/env bash
# Downloads WebGraph (Boldi and Vigna's reference HyperBall) and its runtime dependencies from
# Maven Central into scripts/bench/webgraph/lib/ (gitignored), for hyperball_webgraph.py.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lib
REPO=https://repo1.maven.org/maven2
# group:artifact:version, as webgraph 3.6.12's pom (and those of dsiutils and sux4j) list them
JARS=(
  it.unimi.dsi:webgraph:3.6.12
  it.unimi.dsi:dsiutils:2.7.4
  it.unimi.dsi:sux4j:5.4.1
  it.unimi.dsi:fastutil:8.5.16
  it.unimi.di:jsap:20210129
  org.slf4j:slf4j-api:2.0.3
  org.slf4j:slf4j-nop:2.0.3
  org.apache.commons:commons-math3:3.6.1
  org.apache.commons:commons-lang3:3.12.0
  org.apache.commons:commons-configuration2:2.11.0
  org.apache.commons:commons-collections4:4.4
  org.apache.commons:commons-text:1.12.0
  commons-logging:commons-logging:1.3.2
  com.google.guava:guava:33.4.0-jre
)
for coordinate in "${JARS[@]}"; do
  IFS=: read -r group artifact version <<<"$coordinate"
  jar="lib/$artifact-$version.jar"
  [ -s "$jar" ] && continue
  echo "fetching $artifact $version"
  curl -fsSL -o "$jar" "$REPO/${group//.//}/$artifact/$version/$artifact-$version.jar"
done
