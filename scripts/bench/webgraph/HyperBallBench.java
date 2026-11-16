// Boldi and Vigna's reference HyperBall (WebGraph's it.unimi.dsi.webgraph.algo.HyperBall), for
// scripts/bench/hyperball_webgraph.py, which compares it with sparx's. Run from source (Java 11+,
// no compilation step), with the jars scripts/bench/webgraph/fetch_jars.sh downloads:
//
//   java -cp 'lib/*' HyperBallBench.java convert <offsets> <targets> <basename>
//       stores the CSR graph of <offsets> (n + 1 little-endian int64s) and <targets> (little-endian
//       int32s, each node's sorted) as a BVGraph <basename>.{graph,offsets,properties}
//   java -cp 'lib/*' HyperBallBench.java run <basename> <log2m> <threads> <seed> <out prefix> [warm-ups]
//       runs HyperBall on the (symmetric) graph until no counter changes, writing each node's
//       estimated sums of distances and of inverse distances, and its ball's estimated size, to
//       <out prefix>.webgraph.{dist,harm,reach}
//       (little-endian float32s, as HyperBall keeps them), and prints a line of JSON: the run's
//       seconds (init() and run(), as sparx's counters' initialisation and iterations) and its
//       iterations; with [warm-ups], after as many runs (other seeds) left untimed, for JIT
//       compiled code
import it.unimi.dsi.fastutil.ints.Int2DoubleFunction;
import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.ImmutableSequentialGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;
import it.unimi.dsi.webgraph.LazyIntIterators;
import it.unimi.dsi.webgraph.NodeIterator;
import it.unimi.dsi.webgraph.algo.HyperBall;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

public class HyperBallBench {
	/** A CSR graph, memory mapped, as a sequential WebGraph graph (all BVGraph.store needs). */
	static final class Csr extends ImmutableSequentialGraph {
		final LongBuffer offsets;
		final IntBuffer targets;
		final int n;
		final long arcs;

		Csr(final String offsetsFile, final String targetsFile) throws IOException {
			offsets = map(offsetsFile).asLongBuffer();
			targets = map(targetsFile).asIntBuffer();
			n = offsets.capacity() - 1;
			arcs = offsets.get(n);
			if (arcs != targets.capacity()) throw new IllegalArgumentException("offsets end at " + arcs + ", not at the " + targets.capacity() + " targets");
		}

		static java.nio.ByteBuffer map(final String file) throws IOException {
			try (FileChannel channel = FileChannel.open(Path.of(file), StandardOpenOption.READ)) {
				// (targets past 2^31 bytes would need several buffers: not for these benchmarks)
				return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.LITTLE_ENDIAN);
			}
		}

		@Override
		public int numNodes() {
			return n;
		}

		@Override
		public long numArcs() {
			return arcs;
		}

		@Override
		public NodeIterator nodeIterator() {
			return new NodeIterator() {
				int next = 0;
				int[] successors = new int[16];
				int degree;

				@Override
				public boolean hasNext() {
					return next < n;
				}

				@Override
				public int nextInt() {
					final int u = next++;
					final int begin = (int)offsets.get(u), end = (int)offsets.get(u + 1);
					degree = end - begin;
					if (successors.length < degree) successors = new int[Math.max(degree, 2 * successors.length)];
					targets.get(begin, successors, 0, degree);
					return u;
				}

				@Override
				public int outdegree() {
					return degree;
				}

				@Override
				public int[] successorArray() {
					return successors;
				}

				@Override
				public LazyIntIterator successors() {
					return LazyIntIterators.wrap(successors, degree);
				}
			};
		}
	}

	static void writeFloats(final String file, final float[] values) throws IOException {
		try (FileChannel channel = FileChannel.open(Path.of(file), StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
			final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
			buffer.asFloatBuffer().put(values);
			while (buffer.hasRemaining()) channel.write(buffer);
		}
	}

	public static void main(final String[] args) throws Exception {
		switch (args.length > 0 ? args[0] : "") {
		case "convert" -> {
			if (args.length != 4) usage();
			BVGraph.store(new Csr(args[1], args[2]), args[3]);
		}
		case "run" -> {
			if (args.length != 6 && args.length != 7) usage();
			final ImmutableGraph graph = BVGraph.load(args[1]);
			final int log2m = Integer.parseInt(args[2]), threads = Integer.parseInt(args[3]);
			final long seed = Long.parseLong(args[4]);
			// runs first discarded, so that the one timed runs JIT compiled code, as a long-lived
			// JVM would (other seeds: their results are thrown away anyway)
			final int warmups = args.length == 7 ? Integer.parseInt(args[6]) : 0;
			for (int w = 0; w < warmups; w++) {
				final HyperBall warm = new HyperBall(graph, graph, log2m, null, threads, 0, 0, false, true, true, new Int2DoubleFunction[0], seed + 1 + w);
				warm.init(seed + 1 + w);
				warm.run();
				warm.close();
			}
			// a symmetric graph is its own transpose: HyperBall's systolic iterations need one
			final long start = System.nanoTime();
			final HyperBall hyperBall = new HyperBall(graph, graph, log2m, null, threads, 0, 0, false, true, true, new Int2DoubleFunction[0], seed);
			hyperBall.init(seed);
			hyperBall.run();
			final double seconds = (System.nanoTime() - start) / 1e9;
			// the neighbourhood function's first value is |V|, then one per iteration
			final int iterations = hyperBall.neighbourhoodFunction.size() - 1;
			// each node's ball's estimated size (reached nodes, itself included)
			final float[] reach = new float[graph.numNodes()];
			for (int u = 0; u < reach.length; u++) reach[u] = (float)hyperBall.count(u);
			hyperBall.close();
			writeFloats(args[5] + ".webgraph.reach", reach);
			writeFloats(args[5] + ".webgraph.dist", hyperBall.sumOfDistances);
			writeFloats(args[5] + ".webgraph.harm", hyperBall.sumOfInverseDistances);
			System.out.printf("{\"impl\": \"webgraph\", \"seconds\": %s, \"iterations\": %d, \"nodes\": %d, \"arcs\": %d}%n", seconds, iterations, graph.numNodes(), graph.numArcs());
		}
		default -> usage();
		}
	}

	static void usage() {
		System.err.println("usage: HyperBallBench convert <offsets> <targets> <basename> | run <basename> <log2m> <threads> <seed> <out prefix> [warm-up runs]");
		System.exit(2);
	}
}
