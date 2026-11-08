// Builds a BVH off the page's main thread: posts back what bvh.build returns.
import { build } from './bvh.js';

self.onmessage = (event) => {
  const bvh = build(event.data);
  self.postMessage(bvh, [bvh.nodes.buffer, bvh.order.buffer]);
};
